def flatten_values(d):
    values = []
    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively collect values from the nested dictionary.
            values.extend(flatten_values(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict):
                    # Recursively collect values from the nested dictionary within the list or tuple.
                    values.extend(flatten_values(item))
                else:
                    # Collect the value directly.
                    values.append(item)
        else:
            # Collect the value directly.
            values.append(value)
    return values

# Example of usage:
nested_dict = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}, 'f': (4, {'g': 5, 'h': 6}), 'i': [7, 8, {'j': 9}]}
flattened_values = flatten_values(nested_dict)
print(flattened_values)
#[1, 2, 3, 4, 5, 6, 7, 8, 9]


def flatten_dictionary(nested_dict, sep):
    result = {}
    for key, value in nested_dict.items():
        if type(value) == dict:
            flatten_value = flatten_dictionary(value, sep)
            for inner_key, inner_value in flatten_value.items():
                new_key = f'{key}{sep}{inner_key}'
                if new_key not in result:
                    result[new_key] = inner_value
        elif str(key) not in result:
            result[str(key)] = value
    return result
nested_dict = {'C': {'S': {'1': {'0': {'1': {'0':'S'}}}}}}
print(flatten_dictionary(nested_dict, '-'))
#{'C-S-1-0-1-0': 'S'}

def power_set(lst):
    arr = []
    subset = []
    def helper(i):
        if i>=len(lst):
            arr.append(subset.copy())
            return
        #include
        subset.append(lst[i])
        helper(i+1)
        #dont include
        subset.pop()
        helper(i+1)
    helper(0)
    return arr

print(power_set([1, 2, 3]))
#[[1, 2, 3], [1, 2], [1, 3], [1], [2, 3], [2], [3], []]


def deep_reverse(lst):
    if type(lst)!=list:  # Base case: not a list, return item as is
        return lst
    if len(lst)==0:  # Base case: empty list, return empty list
        return lst
    return deep_reverse(lst[1:]) + [deep_reverse(lst[0])]


def deep_sum(lst):
    if type(lst)==int or type(lst)==float:
        return lst
    
    if len(lst)==0:
        return 0
    
    return deep_sum(lst[0]) + deep_sum(lst[1:])



def count_k_ways(n, k):
    if n<0:
        return 0
    if n<=1:
        return 1
    else:
        return sum(count_k_ways(n-i, k) for i in range(1,k+1))



#top k
#SORT LIST FIRST
if k < len(ans) and k > 0:
    kth = ans[k-1][1]
while k < len(ans) and ans[k][1] == kth:
    k += 1
return ans[:k]

#Finding max of dictionary:
d[month] = max(locs.items(), key=lambda x:x[1])


#Pascal triangle
def pascal(n):
    if n ==1:
        return ((1,),)
    else:
        new_row = (1,)
        prev_row = pascal(n-1)[-1]
        for j in range(0,n-2,1):
            new_row += (prev_row[j] + prev_row[j+1],)
        new_row += (1,)
        return  pascal(n-1) + (new_row,)



#Count number of occurences:
def count_occurrences(lst, num):
    if type(lst) is int:
        if lst == num:
            return 1
        else:
            return 0
    else:
        count = 0
        for el in lst:
            if type(el) is int:
                if el == num:
                    count += 1
            else:
                count = count + count_occurrences(el,num)
        return count

#count_occurrences([1, [2, 1], 1, [3, [1, 3]], [4, [1], 5], [1], 1, [[1]]], 1) == 8




#Prefix infix question
def prefix_infix(expr):
    if isinstance(expr, int):
        return str(expr)
    if not isinstance(expr, list) or len(expr) < 3:
        return str(expr) 
        
    operator = expr[0]
    left_operand = expr[1]
    right_operand = expr[2]
    if isinstance(left_operand, list):
        left_str = prefix_infix(left_operand) 
    else:
        left_str = str(left_operand)  
        
    if isinstance(right_operand, list):
        right_str = prefix_infix(right_operand)  
    else:
        right_str = str(right_operand)  
    return f"({left_str} {operator} {right_str})"



#Coin change:
def num_of_possible_path(board) :
    if len(board) == 2:
        return 1
    if len(board) == 3:
        return 2
    return num_of_possible_path(board[0:-1]) + num_of_possible_path(board[0: -2])



#Slides and ladders problem
def num_plays(pos, board):
    if pos < 1:
        return 0
    elif pos >= board[0]:
          return 1
    else:
        if (pos in board[2]):
            # player is on top of a chute
            return 0
        elif (pos in board[1]):
            # player is at the bottom of a ladder
            pos = board[1][pos]
        res = 0
        for moves in range(1, 7):
            res += num_plays(pos + moves, board)
        return res


































































































































































