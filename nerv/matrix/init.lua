function nerv.Matrix:__tostring__()
    local ncol = self:ncol()
    local nrow = self:nrow()
    local dim = self:dim()
    local strt = {}
    local fmt
    if self.fmt then
        fmt = self.fmt
    else
        fmt = "%.8f "
    end
    if (dim == 2) then
        for row = 0, nrow - 1 do
            local rp = self[row]
            for col = 0, ncol - 1 do
                table.insert(strt, string.format(fmt, rp[col]))
            end
            table.insert(strt, "\n")
        end
    else
        for col = 0, ncol - 1 do
            table.insert(strt, string.format(fmt, self[col]))
        end
        table.insert(strt, "\n")
    end
    table.insert(strt, string.format(
        "[%s %d x %d]", self.__typename, nrow, ncol))
    return table.concat(strt)
end

-- gen: a function takes take indices of the matrix and return the generated
-- all entrys in the matrix will be assigned by calling gen(i, j), for a vector, gen(j) will be called.
function nerv.Matrix:generate(gen)
    if (self:dim() == 2) then
        for i = 0, self:nrow() - 1 do
            local row = self[i]
            for j = 0, self:ncol() - 1 do
                row[j] = gen(i, j)
            end
        end
    else
        for j = 0, self:ncol() - 1 do
            self[j] = gen(j) 
        end
    end
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
    c = nerv.get_type(self.__typename)(self:nrow(), b:ncol())
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
