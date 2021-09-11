#!/usr/bin/env python3
# Implementing Algorithm 1 from the tech report: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2021/EECS-2021-10.pdf
# paper (seems very different from tech report): https://arxiv.org/pdf/2107.13477
# Advanced tutorial (Z3): https://ece.uwaterloo.ca/~agurfink/stqam/z3py-advanced
# Reference manual (Z3): https://z3prover.github.io/api/html/namespacez3py.html#ab8fb082f1c350596ec57ea4a4cd852fd
from z3 import *

def smt(e): 
    s = Solver()
    s.add([e])
    return s

# https://stackoverflow.com/a/12600208/5305365 
def z3_to_py(v):
    if is_bool(v):
        return is_true(v)
    if is_int_value(v):
        return v.as_long()
    raise RuntimeError("unknown z3 value to be coerced |%s|" % (v, ))

#thetai: uninterpreted function
#Ii: oracle function
#rho: expression
#model: SMT model
def consistent(thetai, Ii, rho, model): 
    alpha_total = True 
    # for app in applications of thetai in rho:
    for app in rho.children():
        if is_app(app) and app.decl() == thetai: 
                model_val = model.eval(app)
                arg_vals_in_model = [z3_to_py(model.eval(arg)) for arg in app.children()]
                oracle_val = Ii(*arg_vals_in_model)
                model_app_val = model.eval(thetai(*app.children()))

                # current constraint
                concrete_app = thetai(*arg_vals_in_model)
                alpha_new = concrete_app == oracle_val
                alpha_total = And(alpha_total, alpha_new)
                # print("oracle_val: %s | appval: %s ~= %s | not-equal? %s" % (oracle_val, model_app_val, z3_to_py(model_app_val), oracle_val != z3_to_py(model_app_val)))
                if oracle_val != z3_to_py(model_app_val):

                    # TODO: consider And(alpha_total, alpha_new)
                    # print("early returning")
                    return False, alpha_new
                # alphacur = 
                # print("app: %s | modelval: %s(%s) = %s | oracleval: %s " % (app, appval , arg_vals_in_model, model_val, oracle_val))
        is_consistent, alpha_new = consistent(thetai, Ii, app, model)
        alpha_total = And(alpha_total, alpha_new)
        if not is_consistent:
            return False, alpha_new
    return True, alpha_total

    # for app in rho:
    #     inputs = evaluate(app, model)
    #     response = call_oracle(Ii, inputs)
    #     if response != evaluate(thetai(inputs), model):
    #         return False, alphanew
    # return True, alphanew

def check(xs, thetas, rho, Is):
    alpha = True
    while True:
        success = True
        s = smt(And(rho,  alpha)) 
        if s.check() == unsat:
            return (False, alpha)
        else:
            model = s.model()
            for i in range(len(thetas)):
                is_consistent, alphanew = consistent(thetas[i], Is[i], rho, model)
                print("is_consistent: %s | alphanew: %s" % (is_consistent, alphanew))
                alpha = And(alpha, alphanew)
                if not is_consistent:
                    success = False
                    break   
            if success:
                return (True, model)


def isPrime(x):
    if x < 2: return False
    if x == 2: return True
    for d in range(2, x//2 + 1):
        if x % d == 0:
            return False
    return True
# U = uninterpreted
isPrimeU = Function('isPrime', IntSort(), BoolSort())
x = Int('x')
y = Int('y')
z = Int('z')
query = And(x * y * z == 76, isPrimeU(x) == True, isPrimeU(y) == True, isPrimeU(z) == True) 

print("****OUTPUT****\n")
success, out = check([x, y, z], [isPrimeU], query, [isPrime])
print("success: %s\nout:%s" % (success, out))

# expr = isPrimeU(x)
# print(expr)
# 
# s = Solver()
# s.add([query])
# out = s.check()
# m = s.model()
# print("out: %s | model: %s" % (out, m))
# 


print("****OUTPUT TWO****\n")

queryFalse = And(x * y * z == 100, isPrimeU(x) == True, isPrimeU(y) == True, isPrimeU(z) == True) 
success, out = check([x, y, z], [isPrimeU], queryFalse, [isPrime])
print("success: %s\nout:%s" % (success, out))



print("****OUTPUT THREE****\n")
queryPositive = And(x * y * z == 385, x >= 2, y >= 2, z >= 2, isPrimeU(x) == True, isPrimeU(y) == True, isPrimeU(z) == True) 
success, out = check([x, y, z], [isPrimeU], queryPositive, [isPrime])
print("success: %s\nout:%s" % (success, out))
