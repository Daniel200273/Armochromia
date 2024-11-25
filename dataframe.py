import colorsys
import pandas as pd
import os
import skincolors
import average_color_of_eyes_samples
import average_color_of_hair_samples
from glob import glob
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

eyes = glob('/Users/daniel/Desktop/Armochromia/eye_colors/*.png')
eye_colors= average_color_of_eyes_samples.get_colors_dict(eyes)

hair = glob('/Users/daniel/Desktop/Armochromia/hair_colors/*.png')
hair_colors = average_color_of_hair_samples.get_colors_dict(hair)

# Dictionary containing RGB values for all seasons, each with 'Skin', 'Hair', and 'Eyes'
season_data = {
    'Light Spring': {
        'Skin': skincolors.get_n_most_frequent_colors(LightSpring),
        'Hair': [hair_colors["pearl_blonde"],hair_colors["strawberry_blonde"],hair_colors["light_golden_blond"],hair_colors["medium_golden_blonde"],hair_colors["golden_honey_blonde"]],
        'Eyes': [eye_colors["light_blue_eyes"],eye_colors["soft_green_eyes"], eye_colors["light_hazel_eyes"],eye_colors["light_brown_eyes"]],
        'Wow': '/Users/daniel/Desktop/Armochromia/wow_colors/light_spring_wow.png'
    },
    'Bright Spring': {
        'Skin': skincolors.get_n_most_frequent_colors(BrightSpring),
        'Hair': [hair_colors["medium_golden_blonde"],hair_colors["golden_honey_blonde"],hair_colors["golden_brown"],hair_colors["medium_aburn"],hair_colors["ebony"]],
        'Eyes': [eye_colors["bright_blue_eyes"],eye_colors["green_eyes"],eye_colors["amber_eyes"],eye_colors["brown_eyes"]],
        'Wow': '/Users/daniel/Desktop/Armochromia/wow_colors/bright_spring_wow.png'
    },
    'Warm Spring': {
        'Skin': skincolors.get_n_most_frequent_colors(WarmSpring),
        'Hair': [hair_colors["golden_blond_with_highlights"],hair_colors["golden_honey_blonde"],hair_colors["natural_ginger"],hair_colors["auburn_strawberry"],hair_colors["warm_cinnamon"]],
        'Eyes': [eye_colors["dazzling_grey_eyes"],eye_colors["warm_green_eyes"],eye_colors["light_hazel_eyes"],eye_colors["amber_eyes"]],
        'Wow': '/Users/daniel/Desktop/Armochromia/wow_colors/warm_spring_wow.png'
    },
    'Light Summer': {
        'Skin': skincolors.get_n_most_frequent_colors(LightSum),
        'Hair': [hair_colors["light_cool_blonde"],hair_colors["cool_rose_blonde"],hair_colors["light_ash_blonde"],hair_colors["sandy_blonde"],hair_colors["medium_ash_blonde"],hair_colors["ash_brown"]],
        'Eyes': [eye_colors["light_grey_eyes"],eye_colors["light_blue_eyes"],eye_colors["green_eyes"],eye_colors["azure_eyes"]],
        'Wow': '/Users/daniel/Desktop/Armochromia/wow_colors/light_summer_wow.png'

    },
    'Muted Summer': {
        'Skin': skincolors.get_n_most_frequent_colors(MutedSum),
        'Hair': [hair_colors["cool_brown"],hair_colors["medium_ash_blonde"],hair_colors["ash_brown"],hair_colors["sandy_blonde"],hair_colors["dark_ash_brown"]],
        'Eyes': [eye_colors["grey_eyes"],eye_colors["bright_blue_eyes"],eye_colors["soft_green_eyes"],eye_colors["light_brown_eyes"]],
        'Wow': '/Users/daniel/Desktop/Armochromia/wow_colors/muted_summer_wow.png'
    },
    'Cool Summer': {
        'Skin': skincolors.get_n_most_frequent_colors(CoolSum),
        'Hair': [hair_colors["snow_white"],hair_colors["medium_ash_blonde"],hair_colors["dark_ash_blonde"],hair_colors["cool_brown"],hair_colors["dark_ash_brown"]],
        'Eyes': [eye_colors["grey_eyes"],eye_colors["cool_blue_eyes"],eye_colors["green_eyes"],eye_colors["amber_eyes"]],
        'Wow': '/Users/daniel/Desktop/Armochromia/wow_colors/cool_summer_wow.png'
    },
    'Dark Autumn': {
        'Skin': skincolors.get_n_most_frequent_colors(DarkAutumn),
        'Hair': [hair_colors["medium_warm_brown"],hair_colors["deep_brown_with_caramel"],hair_colors["dark_chestnut"],hair_colors["deep_golden_brown"],hair_colors["mocha_brown"],hair_colors["ebony"]],
        'Eyes': [eye_colors["deep_brown_eyes"],eye_colors["dark_green_eyes"],eye_colors["deep_rosy_brown_eyes"],eye_colors["deep_espresso_eyes"]],
        'Wow': '/Users/daniel/Desktop/Armochromia/wow_colors/dark_autumn_wow.png'
    },
    'Muted Autumn': {
        'Skin':skincolors.get_n_most_frequent_colors(MutedAutumn),
        'Hair': [hair_colors["champagne_blonde"],hair_colors["soft_honey_blonde"],hair_colors["golden_honey_blonde"],hair_colors["caramel_blonde"],hair_colors["soft_honey_brown"],hair_colors["mushroom_brown"]],
        'Eyes': [eye_colors["cool_blue_eyes"],eye_colors["soft_green_eyes"],eye_colors["light_hazel_eyes"],eye_colors["light_brown_eyes"]],
        'Wow': '/Users/daniel/Desktop/Armochromia/wow_colors/muted_autumn_wow.png'
    },
    'Warm Autumn': {
        'Skin': skincolors.get_n_most_frequent_colors(WarmAutumn),
        'Hair': [hair_colors["golden_honey_blonde"],hair_colors["light_golden_aburn"],hair_colors["light_aburn"],hair_colors["medium_aburn"],hair_colors["ombre_ginger"],hair_colors["rich_and_deep_mahogany"]],
        'Eyes': [eye_colors["amber_eyes"],eye_colors["brown_eyes"],eye_colors["olive_green_eyes"],eye_colors["light_hazel_eyes"]],
        'Wow': '/Users/daniel/Desktop/Armochromia/wow_colors/warm_autumn_wow.png'
        
    },
    'Dark Winter': {
        'Skin': skincolors.get_n_most_frequent_colors(DarkWin),
        'Hair': [hair_colors["bitter_chocolate"],hair_colors["medium_cool_brown"],hair_colors["cool_and_darkest_mahogany"],hair_colors["plum_black"],hair_colors["ebony"],hair_colors["pure_black"]],
        'Eyes': [eye_colors["deep_espresso_eyes"],eye_colors["dark_olive_eyes"],eye_colors["deep_brown_eyes"]],
        'Wow': '/Users/daniel/Desktop/Armochromia/wow_colors/dark_winter_wow.png'

    },
    'Bright Winter': {
        'Skin': skincolors.get_n_most_frequent_colors(BrightWin),
        'Hair': [hair_colors["neutral_brown"],hair_colors["medium_cool_brown"],hair_colors["bitter_chocolate"],hair_colors["ebony"],hair_colors["pure_black"]],
        'Eyes': [eye_colors["bright_cyan_eyes"],eye_colors["emerald_eyes"],eye_colors["dazzling_grey_eyes"],eye_colors["deep_rosy_brown_eyes"]],
        'Wow': '/Users/daniel/Desktop/Armochromia/wow_colors/bright_winter_wow.png'
    },
    'Cool Winter': {
        'Skin': skincolors.get_n_most_frequent_colors(CoolWin),
        'Hair': [hair_colors["platinum_blonde"],hair_colors["silver"],hair_colors["light_ash_grey"],hair_colors["cool_brown"]],
        'Eyes': [eye_colors["light_blue_eyes"],eye_colors["light_hazel_eyes"],eye_colors["steel_grey_eyes"],eye_colors["deep_rosy_brown_eyes"]],
        'Wow': '/Users/daniel/Desktop/Armochromia/wow_colors/cool_winter_wow.png'
    }
}







