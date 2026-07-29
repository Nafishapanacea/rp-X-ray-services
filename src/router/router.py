import torch
from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import tempfile
from PIL import Image
import json
import os
import glob
from zipfile import ZipFile

import asyncio
from concurrent.futures import ThreadPoolExecutor

from model.na_model import load_na_model
from model.tb_model import load_tb_model
from model.disease_classification_model import load_disease_classification_model
from utils.utils import preprocess_image, na_predict, tb_predict, disease_classify, dicom_to_image, prepare_image, extract_dicom_metadata
from fastapi.responses import JSONResponse

from src.configuration.config import DICOM_TEMP_PATH, outputDir, VIEW_MAP, SEX_MAP, DEVICE, LABELS

router = APIRouter()

na_model = load_na_model()
tb_model = load_tb_model()
disease_model = load_disease_classification_model()

executor = ThreadPoolExecutor(max_workers=2)

@router.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
):

    temp_dir = tempfile.mkdtemp(dir=DICOM_TEMP_PATH)

    try:
        # ── 1. Save uploaded ZIP ─────────────────────────────────────────
        temp_file = os.path.join(temp_dir, file.filename)
        with open(temp_file, "wb") as out_file:
            out_file.write(await file.read())

        # ── 2. Extract ZIP ───────────────────────────────────────────────
        with ZipFile(temp_file, "r") as zip_ref:
            root_dir = zip_ref.namelist()[0].split("/")[0]
            zip_ref.extractall(temp_dir)

        root_dir_path = os.path.join(temp_dir, root_dir)

        # ── 3. Find first DICOM file ─────────────────────────────────────
        file_paths  = glob.glob(root_dir_path + "/**/*.dcm", recursive=True)
        file_paths += glob.glob(os.path.join(temp_dir, "**", "*.dicom"), recursive=True)

        if not file_paths:
            return JSONResponse(
                status_code=400,
                content={"error": "No .dcm file found inside the ZIP."}
            )

        dicom_path  = file_paths[0]
        print("DICOM FILE:", dicom_path)
        output_path = os.path.splitext(dicom_path)[0] + ".png"

        # ── 4. DICOM → PNG ───────────────────────────────────────────────
        dicom_to_image(dicom_path, output_path, format="png")

        # ── 5. Preprocess image and metadata ──────────────────────────────────────────
        tensor = preprocess_image(output_path)

        sex, view = extract_dicom_metadata(dicom_path)

        response = {}
        run_abnormal_pipeline = True

        if sex != '' and view != '':
            sex  = sex.capitalize()
            view_tensor = torch.tensor([VIEW_MAP[view]], dtype=torch.long).to(DEVICE)
            sex_tensor  = torch.tensor([SEX_MAP[sex]], dtype=torch.long).to(DEVICE)
        
            na_response = na_predict(na_model, tensor, view_tensor, sex_tensor)

            print("NA Prediction:", na_response)

            if na_response == "Normal":
                print("Image classified as Normal. Skipping TB and disease classification.")
                response   = {"finding": na_response}
                run_abnormal_pipeline = False
        
        if run_abnormal_pipeline:
            print("Image classified as Abnormal. Proceeding with TB and disease classification.")
            # tb_response = tb_predict(tb_model, tensor)
            # image = Image.open(output_path)
            # img_tensor2 = prepare_image(image)
            # diseases = disease_classify(disease_model, img_tensor2, LABELS)

            loop = asyncio.get_event_loop()

            image = Image.open(output_path)
            img_tensor2 = prepare_image(image)

            tb_future = loop.run_in_executor(
                executor, tb_predict, tb_model, tensor
            )   

            disease_future = loop.run_in_executor(
                executor, disease_classify, disease_model, img_tensor2, LABELS
            )

            tb_response, disease_output = await asyncio.gather(tb_future, disease_future)

            disease_labels = disease_output["labels"]
            disease_probs = disease_output["probs"]
            tb_response, disease_output  = await asyncio.gather(tb_future, disease_future)

            disease_labels = disease_output["labels"]
            disease_probs = disease_output["probs"]
            
            if tb_response == "TB Negative" and len(disease_labels) == 0:
                response = {"finding": "Normal"}
            else:
                response = {"finding": "Abnormal"}
                response["tb_prediction"] = tb_response
                response["diseases"] = disease_probs

        # ── 6. Save finding to predictions.json ──────────────────────────
        json_filepath = os.path.join(outputDir, "predictions.json")

        with open(json_filepath, "w", encoding="utf-8") as json_file:
            json.dump(response, json_file, indent=4)

        # ── 7. Move JSON to fresh temp dir, return file_id ───────────────
        temp_dir_return = tempfile.mkdtemp(dir=DICOM_TEMP_PATH)
        shutil.move(json_filepath, temp_dir_return)

        return JSONResponse(content={
            "file_id": os.path.basename(temp_dir_return)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    finally:
        # ── Clean up upload temp dir ──────────────────────────────────────
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/json/{file_id}")
async def get_json_object(file_id: str):
    path     = os.path.join(DICOM_TEMP_PATH, file_id, "predictions.json")
    dir_path = os.path.join(DICOM_TEMP_PATH, file_id)

    try:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return data
        else:
            raise HTTPException(status_code=404, detail="File not found")

    except Exception as e:
        raise e

    finally:
        # Clean up temp dir after reading
        if os.path.exists(path):
            os.remove(path)
        if os.path.exists(dir_path) and not os.listdir(dir_path):
            os.rmdir(dir_path)