
from keras import models

model = models.load_model('saved_model/my_model')

def sin(num):
    return (model.predict([1.75])[0] * 2) - 1


#Enter number here
answer = sin(1.75)
print(answer)
