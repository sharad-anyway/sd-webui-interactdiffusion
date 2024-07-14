
from train_pipeline import pipeline
from ldm.modules.diffusionmodules.openaimodel import UNetModel

if __name__ == "__main":

    pipe = pipeline()

    boxes = [(1,1,2,2),(2,2,3,3)]
    grounding_texts = "person=lifting=pallet"
    enabled = True
    stage_one = 0.7
    stage_two = 0.7
    strength = 0.8
    
    # trained_model = pipe.train_dreambooth()

    trained_model = UNetModel

    pipe.process(boxes,grounding_texts,enabled,stage_one,stage_two,strength,trained_model)