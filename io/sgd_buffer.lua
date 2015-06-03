local SGDBuffer = nerv.class("nerv.SGDBuffer", "nerv.DataBuffer")

function SGDBuffer:__init(global_conf, buffer_conf)
    self.gconf = global_conf
    self.buffer_size = math.floor(buffer_conf.buffer_size /
                                global_conf.batch_size) * global_conf.batch_size
    self.head = 0
    self.tail = 0
    self.readers = {}
    for i, reader_spec in ipairs(buffer_conf.readers) do
        local buffs = {}
        for id, width in pairs(reader_spec.data) do
            buffs[id] = {data = global_conf.mmat_type(self.buffer_size, width),
                        leftover = {},
                        width = width}
        end
        table.insert(self.readers, {buffs = buffs,
                                    reader = reader_spec.reader,
                                    tail = 0,
                                    has_leftover = false})
    end
end

function SGDBuffer:saturate()
    local buffer_size = self.buffer_size
    self.head = 0
    self.tail = buffer_size
    for i, reader in ipairs(self.readers) do
        reader.tail = 0
        if reader.has_leftover then
            local lrow
            for id, buff in pairs(reader.buffs) do
                lrow = buff.leftover:nrow()
                if lrow > buffer_size then
                    nerv.error("buffer size is too small to contain leftovers")
                end
                buff.data:copy_from(buff.leftover, 0, lrow)
            end
            reader.tail = lrow
            reader.has_leftover = false
        end
        while reader.tail < buffer_size do
            local data = reader.reader:get_data()
            if data == nil then
                break
            end
            local drow = nil
            for id, d in pairs(data) do
                if drow == nil then
                    drow = d:nrow()
                elseif d:nrow() ~= drow then
                    nerv.error("reader provides with inconsistent rows of data")
                end
            end
            local remain = buffer_size - reader.tail
            if drow > remain then
                for id, buff in pairs(reader.buffs) do
                    local d = data[id]
                    if d == nil then
                        nerv.error("reader does not provide data for %s", id)
                    end
                    buff.leftover = self.gconf.mmat_type(drow - remain,
                                                        buff.width)
                    buff.leftover:copy_from(d, remain, drow)
                end
                drow = remain
                reader.has_leftover = true
            end
            for id, buff in pairs(reader.buffs) do
                buff.data:copy_from(data[id], 0, drow, reader.tail)
            end
            reader.tail = reader.tail + drow
        end
        self.tail = math.min(self.tail, reader.tail)
    end
    return self.tail >= self.gconf.batch_size
end

function SGDBuffer:get_data()
    local batch_size = self.gconf.batch_size
    if self.head >= self.tail then -- buffer is empty
        if not self:saturate() then
            return nil -- the remaining data cannot build a batch
        end
    end
    if self.head + batch_size > self.tail then
        return nil -- the remaining data cannot build a batch
    end
    local res = {}
    for i, reader in ipairs(self.readers) do
        for id, buff in pairs(reader.buffs) do
            local batch = self.gconf.cumat_type(batch_size, buff.width)
            batch:copy_fromh(buff.data, self.head, self.head + batch_size)
            res[id] = batch
        end
    end
    self.head = self.head + batch_size
    return res
end
