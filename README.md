# Technical test

## Data exploration
I start by checking the dataset and see the data I have.
The available data is (example):
- 'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=448x448>,
- 'deepcad_id': '0000/00006371',
- 'cadquery': 'import cadquery as cq\n# Generating a workplane for sketch 0\nwp_sketch0 = cq.Workplane(cq.Plane(cq.Vector(-0.015625, -0.0078125, 0.0), cq.Vector(1.0, 0.0, 0.0), cq.Vector(0.0, 0.0, 1.0)))\nloop0=wp_sketch0.moveTo(0.0, 0.0).threePointArc((0.0007948582418457166, -0.0019189575476279677), (0.0027138157894736844, -0.0027138157894736844)).lineTo(0.021217105263157895, -0.0027138157894736844).threePointArc((0.022787161438489866, -0.00206347722796355), (0.0234375, -0.000493421052631579)).lineTo(0.0234375, 0.018256578947368422).threePointArc((0.02283825686147997, 0.019949990385858287), (0.021217105263157895, 0.020723684210526318)).lineTo(0.0022203947368421052, 0.020723684210526318).threePointArc((0.0005992431385200307, 0.019949990385858287), (0.0, 0.018256578947368422)).lineTo(0.0, 0.0).close()\nsolid0=wp_sketch0.add(loop0).extrude(0.75)\nsolid=solid0\n',
- 'token_count': 1292,
- 'prompt': 'Generate the CADQuery code needed to create the CAD for the provided image. Just the code, no other words.',
- 'hundred_subset': False

## Thinking process
I want to build a model to generate CadQuery code 
Input: image
Output: cadquery (text)

I will use Resnet18 to convert the image into a vector and then an embedding as a decoder model.
I implemented a dataset format that allows me to have the image, the vector and the query for each element of the initial dataset

I trained the model during 2 epochs. It is quite slow
The loss is around 0.2 using the entire dataset.
I implemented another function to split the dataset into test, validation and train


I had not find the time to test with the given evaluation metrics.