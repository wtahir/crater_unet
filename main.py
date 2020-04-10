from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/crater/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='loss',verbose=1, save_best_only=False)
model.fit_generator(myGene,steps_per_epoch=208,epochs=3,callbacks=[model_checkpoint])

testGene = testGenerator("data/crater/test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/crater/test",results)
