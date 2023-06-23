function write_markdown()
    jacobian_header = "| Function | FD | Enzyme | ForwardDiff | ReverseDiff |"
    table_vals = run_hessianbenchmarks()

