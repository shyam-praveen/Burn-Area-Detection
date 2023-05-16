import numpy as np
import base64
import os
import time
from typing import Annotated
from fastapi import FastAPI, File, UploadFile, Body, Request
import cv2
import sys
from visualize import random_colors, get_mask_contours, draw_mask   
from m_rcnn import *
import io
from pydantic import BaseModel

class TextIn(BaseModel):
    text: str

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

body_model, inference_config = load_inference_model(1, "BodyModel.h5")
burn_model, inference_config = load_inference_model(1, "BurnModel.h5")

def helper(image):
    # img = cv2.imread("1.jpeg")
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictionBody = body_model.detect([image])[0]
    predictionBurn = burn_model.detect([image])[0]
    ansBurn = 0
    ansBody = 0

    if predictionBody["masks"].shape[2] != 0:
        ansBody = np.count_nonzero(predictionBody["masks"][:, :, 0])
    if predictionBurn['masks'].shape[2] != 0:
        ansBurn = np.count_nonzero(predictionBurn["masks"][:, :, 0])
    
    if ansBody != 0:
        return (ansBurn/ansBody)*100
    else:
        return -1

# Create the FastAPI app
app = FastAPI()

# Define an endpoint at the root URL
@app.get("/")
async def root():
    return {"API": "Home Route of Burn area percentage calculation API"}


l = []
ans = 0
fluid = 0
times = 0
@app.post("/predict/upload")
async def create_upload_file(file: Request):
    a = await file.json()
    decoded_image_data = base64.b64decode(a['image'])
    np_array = np.frombuffer(decoded_image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    global ans

    ans += helper(image)
    # print(image)
    # ans = 0
    print(ans)
    return ans

@app.get("/predict/endpoint")
async def root():
    global ans
    print(ans)
        
    return ans

@app.get("/predict/finalResult/endpoint")
async def root():
    global ans
    print(ans/2)
    finalBurnPercentage = ans/2
    ans = 0

    if finalBurnPercentage > 100:
        return 70

    return finalBurnPercentage

@app.post("/predict/form")
async def formFill(file: Request):
    val = await file.json()
    # val = file
    print(val)
    names = ""
    weight = 0
    k = ""
    f = 1
    s = val['int1']

    for i in range(len(s)):
        if s[i] != '#':
            k += s[i]

        if s[i] == '#' and f == 1:
            names = k
            k = ""
            f = 2
        elif s[i] == '#' and f == 2:
            weight = int(k)
            k = ""
    
    global times
    times = int(k)

    print(names)
    print(times)
    print(weight)
    global fluid
    fluid = 4*weight*ans
    return fluid

@app.get("/predict/fluid")
async def fluidFinal():
    global fluid
    val = fluid
    print(val)
    fluid = 0
    global times 
    s = "Fluid to be given in next "
    s += str(480 - times)
    s += " = "
    s += str(val/2)
    s += "ml. Fluid to be given in next "
    s += str(480 + 960 - times)
    s += " = "
    s += str(val/2)
    s += "ml"
    s = f'Total amount of fluid to be given {val}ml'
    
    print(s)
    return s