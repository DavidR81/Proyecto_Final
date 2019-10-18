from PIL import Image
from keras.models import model_from_json
from PIL import Image
import numpy as np

path = "../Pruebas/ciudad.jpg"
def cambioImagen(path):
    im1 = Image.open(path)
    width = 64
    height = 64
    im2 = im1.resize((width, height), Image.BILINEAR)
    ext = ".jpg"
    im2.save("test" + ext)
    return im2


json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
#print("Loaded model from disk")


clases = {0 : 'AnnualCrop',
          1 : 'Forest',
          2 : 'HerbaceousVegetation',
          3 : 'Highway',
          4 : 'Industrial',
          5 : 'Pasture',
          6 : 'PermanentCrop',
          7 : 'Residential',
          8 : 'River',
          9 : 'SeaLake'}


img=np.array(Image.open('../SRC/test.jpg'))
img=np.array([img/255])
#print(np.max(img))

pred=loaded_model.predict(img)
pred=list(pred[0])
resultado = pred.index(max(pred))
resultado
print('El tipo de terreno es:', clases[resultado])

urbanizable = ['AnnualCrop', 'Residential', 'HerbaceousVegetation', 'Industrial', 'Residential']

    
if clases[resultado] in urbanizable:
    print('Es urbanizable')
else:
    print('El terreno no es urbanizable')




