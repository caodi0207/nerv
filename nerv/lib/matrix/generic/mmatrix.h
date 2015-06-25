#include "../../common.h"
#include "../../io/chunk_file.h"

Matrix *nerv_matrix_(load)(ChunkData *cdp, Status *status);
void nerv_matrix_(save)(Matrix *self, ChunkFile *cfp, Status *status);
void nerv_matrix_(copy_from)(Matrix *a, const Matrix *b,
                            int a_begin, int b_begin, int b_end,
                            Status *status);
