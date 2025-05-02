# from datasets import load_dataset
# import io
# import matplotlib.pyplot as plt
# import PIL
# from PIL import Image

#dataset = load_dataset("array/SAT", batch_size=128, cache_dir="/projectnb/cs598/students/achetia")
# dataset should have a training and validation key

# example = dataset['train'][15] # example 10th item

# image = example['image'] # this is a list of images. Some questions are on one image, and some on 2 images
# question = example['question']
# answer_choices = example['choices']
# correct_answer = example['answer']

# print(f"Question: {question}")
# for idx, choice in enumerate(answer_choices):
#     print(f"{idx + 1}: {choice}")

# print(f"Correct Answer: {correct_answer}")

# image.show()