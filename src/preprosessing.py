
import os

CATEGORIES = [
    'dyed-lifted-polyps', 
    'dyed-resection-margins', 
    'esophagitis', 
    'normal-cecum', 
    'normal-pylorus', 
    'normal-z-line',
    'polyps', 
    'ulcerative-colitis'
]


for category in CATEGORIES:
    path = os.path.join('data/', category)
    class_num = CATEGORIES.index(category)
    i = 0

    for filename in os.listdir(path): 
        dst =category + str(i) + ".jpg"   
        src =path + '/'+ filename 

        if i < 900: 
            dst = 'data/train/'+category+'/'+ dst 
        else:
            dst = 'data/test/'+category+'/'+ dst 
        os.rename(src, dst) 
        i += 1