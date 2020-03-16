# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:27:41 2020
created for python practice from cognitiveclass.ai website
@author: rachi
"""
#first python program
'''display python 101'''
print("Hello Python 101 course")

# Check the Python Version
import sys
print(sys.version)

#types
'''dsiplay various data types in python'''
type(11) #int data type
type(11.123) #float data type
type("Hello python") #string data type
type(True) #boolean data type
type(False) #boolean data type

#type casting
float(2) #int to float. output-2.0
int(1.1) #float to int. output-1
int('1') #string to int. output-1
int('A') #string to int. output-ValueError: invalid literal for int() with base 10: 'A'
str(1) #int to string. output-'1'
str(4.5) #float to string. output-'4.5'
int(True) #boolean to int. output-1
int(False) #boolean to int. output-0
float(False) #boolean to int. output-0.0
float(True) #boolean to int. output-1.0
bool(1) #int to boolean. output-True
bool(0) #int to boolean. output-False

#division
25/4 #output-6.25
25//4 #output-6

#string operations
name="Micheal Jackson"
name[0:4] #output-'Mich'
name[8:12] #output-'jack'
name[-1] #output-'n'
name[::2] #output-'McelJcsn'  #every second variable is selected
name[0:5:2] #output-'Mce' #every second variable is selected till 5th position
len(name) #output-15 #calculates the length of string
3*name #output-'Micheal JacksonMicheal JacksonMicheal Jackson'
name[1]='j' #string immutable #TypeError: 'str' object does not support item assignment
name = name+"is the best" #this is possible output-'Micheal Jacksonis the best'
#string escape sequences \n-new line, \t-tab space, \-escape character or use r in front of the string to escape
name_upper = name.upper() #converts to sentence upper case
name_upper = name_upper.replace('MICHEAL','Janet') #replaces the word with the given word
name.find('al') #output-5. #if output is not present then output is -1


#tuples- properties same as strings
'''tuples are ordered sequence.'''
#examples given below-
ratings=(1,2,3,4,5,6,7,8,9,0) 
tuple1=('disco',10,1.2) #can contain string, integer and float
tuple1[0]
tuple1[-1]
tuple1[0:3]
tuple1[2:3] #known as slicing
len(tuple1)
tuple1 = sorted(tuple1) #used for sorting the tuple
#tuples are immutable like strings


#list
'''ordered sequence, mutable, represented using square brackets []'''
list1 = ["abc",2,3,13.2]
list1.extend(["pop"]) #output-['abc', 2, 3, 13.2, 'pop']
list1.append(["aba",1,2.2]) #output-['abc', 2, 3, 13.2, 'pop', ['aba', 1, 2.2]]
del(list1[0])
"hard rosk".split() #split method is used to convert a string to list, all words are converted to individual elements
'''When we set one variable, B equal to A, both A and B are referencing the same list.
Multiple names referring to the same object is known as aliasing.
We know from the last slide that the first element in B is set as hard rock.
If we change the first element in “A” to “banana” we get a side effect; the
value of B will change as a consequence.
"A" and “B” are referencing the same list, therefore if we change "A“, list "B" also
changes.
If we check the first element of B after changing list ”A” we get banana instead of hard
rock.
You can clone list “A” by using the following syntax.'''
a = ["hard rock",10,1.2]
b= a[:]
#now if you change anything in list b then lit a wont get affected
a[2:4] #known as slicing


#set
'''unique ordered sequence'''
#convert list to a set
set1 = set(list1)
set1.add('apple')
set1.remove('apple')


#dictionaries
'''key value pairs, keys are unique and immutable, values can be immutable/mutable,duplicate,  curly brackets are used'''
dict["enter_key_here"] #output will be the corresponding value
dict.keys() #gives all the keys
dict.values() #gives all the values