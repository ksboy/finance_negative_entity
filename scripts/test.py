import torch

a = torch.tensor([[1.,1.,1.,1.],
                 [5.,6.,7.,8.]])
b = torch.tensor([3.,3.,3.,3.])
c = torch.tensor([5.,5.,5.,5.])

list1 = [a, b, c]

print(a.size())
zeros = torch.zeros(6)
print(zeros)
a[0] = torch.cat((a[0],zeros),0)
print(a)
# print(torch.stack((a,b),0))
# print(torch.stack((a,b),0).size())