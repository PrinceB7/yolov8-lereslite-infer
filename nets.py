import cv2
import numpy as np
import onnxruntime

class ONNXSegmentModel():
    """ONNX wrapper for LeResSegNet
    """    
    def __init__(self, checkpoint_path, recover_original=True):
        self.initialize_model(checkpoint_path)
        self.recover_original = recover_original
    
    def initialize_model(self, checkpoint_path):
        self.session = onnxruntime.InferenceSession(checkpoint_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])                                                                               
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        self.output_shape = model_outputs[0].shape
        self.output_height = self.output_shape[2]
        self.output_width = self.output_shape[3] 
        
    def preprocess(self, input_image):
        need_swap = True
        h, w = input_image.shape[:-1]
        input_image = cv2.dnn.blobFromImage(input_image, 1 / 255., (self.input_width, self.input_height), 0, need_swap, False)        
        return input_image, h, w, self.input_height, self.input_width
        
    def predict(self, input_tensor):
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
    
    def postprocess(self, outputs, orginal_width, original_height):
        outputs = outputs.squeeze() #(C, H, W)
        outputs = np.argmax(outputs, axis=0) #(H, W)
        outputs = outputs.astype(np.float32)
        if self.recover_original:
            outputs = cv2.resize(outputs, (orginal_width, original_height), cv2.INTER_NEAREST_EXACT)
        outputs = outputs[None] #(1, H, W)
        return outputs.astype(np.int64)
    
    def estimate_segment(self, image):#
        input_tensor, original_height, original_width, out_h, out_w = self.preprocess(image)
        output_tensor = self.predict(input_tensor)
        outputs = self.postprocess(output_tensor, original_width, original_height)
        return outputs, out_h, out_w
    
    def __call__(self, image):
        outputs, out_h, out_w = self.estimate_segment(image)
        return outputs, out_h, out_w