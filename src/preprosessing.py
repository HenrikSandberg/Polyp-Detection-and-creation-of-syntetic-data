
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

def uniform_naming_categories():
    for category in CATEGORIES:
        path = os.path.join('data/', category)
        i = 0

        for filename in os.listdir(path): 
            dst =category + str(i) + ".jpg"   
            src =path + '/'+ filename 
            dst = 'data/'+category+'/'+ dst 
            os.rename(src, dst) 
            i += 1

def uniform_naming_syntetic():
    for category in CATEGORIES:
        path = os.path.join('data/syntetic/', category)
        i = 0

        for filename in os.listdir(path): 
            dst =category + str(i) + ".jpg"   
            src =path + '/'+ filename 
            dst = 'data/'+category+'/'+ dst 
            os.rename(src, dst) 
            i += 1