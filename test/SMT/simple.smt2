(declare-fun y () Int)
 
(assert (= x y))
(check-sat)
(get-model)