function nerv.FloatCuMatrix:__tostring__()
    local ncol = self:ncol()
    local nrow = self:nrow()
    local strt = {}

    if nrow == 1 then
        for col = 0, ncol - 1 do
            table.insert(strt, string.format("%f ", self[col]))
        end
        table.insert(strt, "\n")
    else
        for row = 0, nrow - 1 do
            local rp = self[row]
            for col = 0, ncol - 1 do
                table.insert(strt, string.format("%f ", rp[col]))
            end
            table.insert(strt, "\n")
        end
    end
    table.insert(strt, string.format("[Float Matrix %d x %d]", nrow, ncol))
    return table.concat(strt)
end

function nerv.FloatMatrix:__tostring__()
    local ncol = self:ncol()
    local nrow = self:nrow()
    local i = 0
    local strt = {}
    for row = 0, nrow - 1 do
        for col = 0, ncol - 1 do
            table.insert(strt, string.format("%f ", self:get_elem(i)))
            i = i + 1
        end
        table.insert(strt, "\n")
    end
    table.insert(strt, string.format("[Float Matrix %d x %d]", nrow, ncol))
    return table.concat(strt)
end
