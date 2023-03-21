def prime(num):
    for i in range(2,num):
        if num % i == 0:
            return 0
    return 1

print(prime(30))
print(prime(13))
print(prime(15))
print(prime(11))
print(prime(17))