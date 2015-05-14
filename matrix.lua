function nerv.FloatMatrix:__tostring__()
    local ncol = self:ncol()
    local nrow = self:nrow()
    local i = 0
    local res = ""
    for row = 0, nrow - 1 do
        for col = 0, ncol - 1 do
            res = res .. string.format("%f ", self:get_elem(i))
            i = i + 1
        end
        res = res .. "\n"
    end
    res = res .. string.format("[Float Matrix %d x %d]", nrow, ncol)
    return res
end
