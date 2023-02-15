import random
def get_colors():
    s='''
        
        #AB63FA, #1CA71C, #1616A7,
        #B00068, #DA16FF, #00CC96,
        #FD3216, #19D3F3, #00FE35, 
        #FF0092, #FEAF16
        
        '''

    li=s.split(',')
    li=[l.replace('\n','') for l in li]
    li=[l.replace(' ','') for l in li]
    random.shuffle(li)
    return li