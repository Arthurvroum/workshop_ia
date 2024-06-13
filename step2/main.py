import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

rock_dir = os.path.join('../../rps/rock')
paper_dir = os.path.join('../../rps/paper')
scissors_dir = os.path.join('../../rps/scissors')

rock_files = os.listdir(rock_dir)

paper_files = os.listdir(paper_dir)

scissors_files = os.listdir(scissors_dir)

pic_index = 2

next_rock = [os.path.join(rock_dir, fname) for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
    # Open each image and plot it on a subplot
    img = mpimg.imread(img_path)
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.axis('Off')  # Turn off axis

plt.show()  # Display the plot