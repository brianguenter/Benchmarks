function fd_product(n)
    x = FastDifferentiation.make_variables(:x, n)
    x_ops = FastDifferentiation.number_of_operations([*(x...)])
    jac = FastDifferentiation.jacobian([*(x...)], x)
    jac_ops = FastDifferentiation.number_of_operations(jac)
    hess = FastDifferentiation.hessian(*(x...), x)
    hess_ops = FastDifferentiation.number_of_operations(hess)
    println("xops $x_ops jac_ops $jac_ops hess $hess_ops")
end
export fd_product