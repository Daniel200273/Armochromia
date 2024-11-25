from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

#n is number of colors in my palette
def get_n_most_frequent_colors(image_path_n):
    image_path= image_path_n[0]
    n=image_path_n[1]
    # Open the image file
    with Image.open(image_path) as img:
        # Ensure the image is in RGB mode
        img = img.convert("RGB")
        
        # Get the dimensions of the image
        width, height = img.size
        
        # Initialize a list to store the colors
        colors = []
    
        
        # Loop through each pixel in the image
        for y in range(height):
            for x in range(width):
                # Get the color of the pixel
                color = img.getpixel((x, y))
                # Add the color to the list
                colors.append(color)
        
        # Use Counter to count the frequency of each color
        color_counts = Counter(colors)
        
        # Get the n most common colors
        most_common_colors = color_counts.most_common(n)
        rgb_tuples = [color for color, count in most_common_colors]
        # Plot the colors
        plt.figure(figsize=(8, 1))
        plt.imshow([rgb_tuples], aspect='auto')
        plt.axis('off')
        plt.show()

        return rgb_tuples
        


#Image Paths and n
LightSpring = ['/Users/daniel/Desktop/Armochromia/final/season_skin_palettes/LightSpringSkin.jpg',6]
BrightSpring = ['/Users/daniel/Desktop/Armochromia/final/season_skin_palettes/BrightSpringSkin.jpg',10]
WarmSpring = ['/Users/daniel/Desktop/Armochromia/final/season_skin_palettes/WarmSpringSkin.jpg',5]
LightSum = ['/Users/daniel/Desktop/Armochromia/final/season_skin_palettes/LightSumSkin.jpg',3]
CoolSum = ['/Users/daniel/Desktop/Armochromia/final/season_skin_palettes/CoolSumSkin.jpg',4]
MutedSum = ['/Users/daniel/Desktop/Armochromia/final/season_skin_palettes/MutedSumSkin.jpg',12]
DarkAutumn = ['/Users/daniel/Desktop/Armochromia/final/season_skin_palettes/DarkAutumnSkin.jpg',15]
WarmAutumn = ['/Users/daniel/Desktop/Armochromia/final/season_skin_palettes/WarmAutumnSkin.jpg',5]
MutedAutumn = ['/Users/daniel/Desktop/Armochromia/final/season_skin_palettes/MutedAutumnSkin.jpg',12]
CoolWin = ['/Users/daniel/Desktop/Armochromia/final/season_skin_palettes/CoolWinSkin.jpg',4]
DarkWin = ['/Users/daniel/Desktop/Armochromia/final/season_skin_palettes/DarkWinSkin.jpg',18]
BrightWin = ['/Users/daniel/Desktop/Armochromia/final/season_skin_palettes/BrightWinSkin.jpg',13]


