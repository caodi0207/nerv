function nerv.Matrix:__tostring__()
    local ncol = self:ncol()
    local nrow = self:nrow()
    local strt = {}
    local fmt
    if self.fmt then
        fmt = self.fmt
    else
        fmt = "%.10f "
    end
    if nrow == 1 then
        for col = 0, ncol - 1 do
            table.insert(strt, string.format(fmt, self[col]))
        end
        table.insert(strt, "\n")
    else
        for row = 0, nrow - 1 do
            local rp = self[row]
            for col = 0, ncol - 1 do
                table.insert(strt, string.format(fmt, rp[col]))
            end
            table.insert(strt, "\n")
        end
    end
    table.insert(strt, string.format("[Matrix %d x %d]", nrow, ncol))
    return table.concat(strt)
end

nerv.MMatrixInt.fmt = "%d "

function nerv.CuMatrix:__add__(b)
    c = self:create()
    c:add(self, b, 1.0, 1.0)
    return c
end

function nerv.CuMatrix:__sub__(b)
    c = self:create()
    c:add(self, b, 1.0, -1.0)
    return c
end

function nerv.CuMatrix:__mul__(b)
    c = self:create()
    c:mul(self, b, 1.0, 0.0, 'N', 'N')
    return c
end

function nerv.CuMatrixFloat.new_from_host(mat)
    local res = nerv.CuMatrixFloat(mat:nrow(), mat:ncol())
    res:copy_fromh(mat)
    return res
end

function nerv.CuMatrixFloat:new_to_host()
    local res = nerv.MMatrixFloat(self:nrow(), self:ncol())
    self:copy_toh(res)
    return res
end
