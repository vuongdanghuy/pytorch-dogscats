import torch
import torchvision
from torchvision.transforms import transforms
import os
import json
import logging
from io import BytesIO
import numpy as np

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_fn(model_dir):
	logger.info('In model_fn')
	logger.info(f'Using device {device}')

	model = torchvision.models.resnet50(pretrained=False)

	model.fc = torch.nn.Sequential(
			torch.nn.Linear(in_features=2048, out_features=1, bias=True),
			torch.nn.Sigmoid()
		)

	logger.info('Loading model...')

	# model_path = os.path.join(model_dir, 'resnet50.pth')
	# with open(model_path, 'rb') as f:
	# 	model.load_state_dict(torch.load(f, map_location=device))

	model_path = f'{model_dir}/model.pth'
	model.load_state_dict(torch.load(model_path, map_location=device))

	logger.info('Loading model done.')

	return model

def _npy_load(data):
	stream = BytesIO(data)
	return np.load(stream)

def _npy_dump(data):
	buffer = BytesIO()
	np.save(buffer, data)
	return buffer.getvalue()


def input_fn(request_body, content_type='application/json'):
	logger.info('In input_fn')
	logger.info('Deserializing the input data.')
	# if content_type == 'application/json':
	# 	input_data = json.loads(request_body)
	# 	logger.info(f'Input data:\n{input_data}')
	# 	url = input_data['url']
	# 	logger.info(f'Image url: {url}')
	# 	image_data = Image.open(requests.get(url, stream=True).raw)

	# 	image_transform = transforms.Compose([
	# 		transforms.ToTensor(),
	# 		transforms.Normalize([0, 0, 0], [1, 1, 1])
	# 	])
	# 	return image_transform(image_data)

	if content_type == 'application/x-npy':
		logger.info(f'Content type is {content_type}.')
		logger.info('Convert request_body to numpy array...')
		image = _npy_load(request_body)

		return torch.from_numpy(image).float()
	elif content_type == 'application/json':
		logger.info(f'Content type is {content_type}')
		input_data = json.loads(request_body)
		image = np.array(input_data['instances'])

		return torch.from_numpy(image).float()

	raise Exception(f'Requested unsupported ContentType in content_type {content_type}')

def predict_fn(input_obj, model):
	"""
	Get prediction
	"""
	logger.info('In predict_fn')

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	with torch.no_grad():
		preds = model(input_obj.to(device))
		logger.info(f'Prediction value: {preds}')
		
		return preds.numpy()
