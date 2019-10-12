import re
import numpy as np

def get_rate(row):
    product1 = row.externalName
    product2 = row.GetirName
    regex = r'[0-9]+'
    numbers1 = set(re.findall(regex, product1))
    numbers2 = set(re.findall(regex, product2))
    union = numbers1.union(numbers2)
    intersection = numbers1.intersection(numbers2)
    if len(numbers1)==0 and len(numbers2) == 0:
        rate = 1
    else:
        rate = (len(intersection)/ len(union))
    return rate


def get_unique_number_count(row):
    product1 = row.externalName
    product2 = row.GetirName
    regex = r'[0-9]+'
    numbers1 = set(re.findall(regex, product1))
    numbers2 = set(re.findall(regex, product2))
    union = numbers1.union(numbers2)
    return len(union)


def levenshteinRecursive(seq1, seq2):

    if seq1 == "":

        return len(seq2)
    if seq2 == "":

        return len(seq1)
    if seq1[-1] == seq2[-1]:
        cost = 0
    else:
        cost = 1

    res = min([levenshteinRecursive(seq1[:-1], seq2) + 1,
               levenshteinRecursive(seq1, seq2[:-1]) + 1,
               levenshteinRecursive(seq1[:-1], seq2[:-1]) + cost])
    return res


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )

    return (matrix[size_x - 1, size_y - 1])


def sorted_levenshtein_apply(row):
    product1 = ''.join(sorted(row.externalName))
    product2 = ''.join(sorted(row.GetirName))
    distance = levenshtein(product1, product2)
    return distance

def sorted_levenshtein_rate_apply(row):
    product1 = ''.join(sorted(row.externalName))
    product2 = ''.join(sorted(row.GetirName))
    distance = levenshtein(product1, product2)
    max_len = max(len(product1), len(product2))
    return 1-(distance/max_len)


def sorted_levenshtein(sequence1, sequence2):
    product1 = ''.join(sorted(sequence1))
    product2 = ''.join(sorted(sequence2))
    distance = levenshtein(product1, product2)
    return distance

def sorted_levenshtein_rate(seq1, seq2):
    product1 = ''.join(sorted(seq1))
    product2 = ''.join(sorted(seq2))
    distance = levenshtein(product1, product2)
    max_len = max(len(product1), len(product2))
    return 1-(distance/max_len)

def levenshtein_rate(product1, product2):
    distance = levenshtein(product1, product2)
    max_len = max(len(product1), len(product2))
    return 1 - (distance / max_len)


if __name__ == "__main__":
    met1 = "Selpak Kağıt Havlu 8'li"
    met2 = "Selpak Kağıt Havlu 12'li"
    print('Levenshtein Distance: {}, MatchScore: {} '.\
          format(levenshtein(met1, met2), levenshtein_rate(met1, met2)))
    print('Sorted Levenshtein Distance: {}, MatchScore: {} '.\
          format(sorted_levenshtein(met1, met2), sorted_levenshtein_rate(met1, met2)))