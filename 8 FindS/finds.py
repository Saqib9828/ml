h=['phi','phi','phi','phi','phi','phi']   


Data=[['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
      ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
      ['Sunny','Warm','Normal','Strong','Warm','Same','No'],
      ['Sunny','Warm','High','Strong','Cool','Change','Yes']
     ]

def isConsistent(h,d):
    
    if len(h)!=len(d)-1:
        print('Number of attributes are not same in hypothesis.')
        return False
    else:

        
        matched=0

       
        for i in range(len(h)):

            
            if ( (h[i]==d[i]) | (h[i]=='any') ):

                
                matched=matched+1

        
        if matched==len(h):
            return True
        else:
            return False


def makeConsistent(h,d):

    
    for i in range(len(h)):

        
        if((h[i] == 'phi')):

            h[i]=d[i]


            
        elif(h[i]!=d[i]):

            h[i]='any'

    return h




print('Begin : Hypothesis :',h)
print('==========================================')
for d in Data:

   
    if d[len(d)-1]=='Yes':

       
        if ( isConsistent(h,d)):

            
            print ("Hypothesis :",d)
        else:

            
            h=makeConsistent(h,d)


        
        print ('Training data         :',d)
        print ('Updated Hypothesis    :',h)
        print()
        print('--------------------------------')

print('==========================================')
print('End: Hypothesis :',h)
