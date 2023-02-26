import numpy as np
import torch
import subprocess
import os
from PIL import Image
import json
import pytesseract
from io import BytesIO
import requests




class OpenALPRBWrapper():
    """
    DEPRECATED. Storing code here in case I need to write something
    similar in the future.
    """
    def __init__(self, imdir, c="eu"):
        """
        :vic_lp:
        :imdir:
        :c: 'eu' or 'us'
        """
        self.imdir = imdir
        self.c = c

    def run_alpr_detect(self, image, return_json=False):
        # PROBABLY NEED TO ADD AN image.permute() to convert from C,H,W to H,W,C
        img_np = image.permute(1,2,0).numpy()
        #img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        #img_name = 'temp.jpg'
        img_name = f'{np.random.randint(0, int(1e9))}.jpg'
        img_path = os.path.join(self.imdir,img_name)
        Image.fromarray((img_np*255).astype(np.uint8)).save(img_path)
        #cv2.imwrite(img_path, img_np * 255.0)
        if return_json:
            a = subprocess.run(["docker", "run", "-i", "--rm", "-v", f"{self.imdir}:/data:ro",
                            "openalpr/openalpr", f"-c {self.c}", "-j", img_name], 
                           stdout=subprocess.PIPE)
        else:
            a = subprocess.run(["docker", "run", "-i", "--rm", "-v", f"{self.imdir}:/data:ro",
                            "openalpr/openalpr", f"-c {self.c}", img_name], 
                           stdout=subprocess.PIPE)
        b = subprocess.run(["rm", img_path])
        if return_json:
            return json.loads(a.stdout.decode('utf-8'))
        else:
            return a.stdout.decode('utf-8')
        
    def detect_plate(self, image, lp):
        """
        Return True if lp is detected
        
        :image: torch Tensor containing an image
        :lp: string containing license plate number (no spaces or hyphens)
        """
        detections = self.run_alpr_detect(image, False)
        return lp in detections

    def predict(self, image, target=None):
        assert False, "not implemented"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            image_batch = image.clone()
            if len(image.size()) < 4:
                image_batch = image_batch.unsqueeze(0)
            predict = torch.zeros((image_batch.size()[0]), dtype=torch.long)
            for i in range(predict.size()[0]):
                lp = self.run_alpr_detect(image_batch[i])
                if lp != self.vic_lp:
                    predict[i] = 1
                else:
                    predict[i] = 0

        predict = predict.to(device)

        if len(image.size()) < 4:
            return predict[0].item()
        return predict
    
    
def quick_and_dirty_lpr(img, diameter=11, sigma=17, cannythreshold1=30, cannythreshold2=200):
    """
    Barebones ALPR pipeline using OpenCV and tesseract.
    """
    # if it's a torch tensor- assume it's in C,H,W format and 
    # a normalized float. convert to a uint8 H,W,C array
    if isinstance(img, torch.Tensor):
        img = (img.permute(1,2,0).numpy()*255).astype(np.uint8)
    # map to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    gray_image_filtered = cv2.bilateralFilter(gray_image, diameter, sigma, sigma)
    # 
    edged = cv2.Canny(gray_image_filtered, cannythreshold1, cannythreshold2)
    #
    cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    #
    quads = []
    for c in cnts_sorted:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018*perimeter, True)
        if len(approx) == 4:
            quads.append(approx)
    text = ""
    for q in quads:
        x,y,w,h = cv2.boundingRect(q)
        new_img = img[y:y+h,x:x+w]
        text += pytesseract.image_to_string(new_img)
    return text




def build_api_detect_function(plate, url='http://localhost:8088/api'):
    """
    
    """
    def detect_function(img):
        memfile = BytesIO()
        img = Image.fromarray((img.permute(1,2,0).numpy()*255).astype(np.uint8))
        img.save(memfile, "JPEG", quality=100)
        memfile.seek(0)
        r = requests.post(url, data=memfile)
        results = json.loads(r.content.decode('utf8'))
        if len(results["results"]) > 0:
            if plate in [x["plate"] for x in results["results"][0]["candidates"]]:
                return 1
            else:
                return 0
        else:
            return 0
    return detect_function