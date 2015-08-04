#include "../matrix.h"
Matrix *nerv_matrix_(create)(long nrow, long ncol, Status *status);
void nerv_matrix_(destroy)(Matrix *self, Status *status);
Matrix *nerv_matrix_(getrow)(Matrix *self, int row);
void nerv_matrix_(data_free)(Matrix *self, Status *status);
void nerv_matrix_(data_retain)(Matrix *self);
