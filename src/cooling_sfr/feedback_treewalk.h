#ifndef FEEDBACK_TREEWALK_H
#define FEEDBACK_TREEWALK_H

#include "../data/dtypes.h"

struct feedback_data_in {
    MyDouble Pos[3];
    MyFloat Mass;
    MyFloat Energy;
    MyFloat MassReturn;
    MyFloat Yield[4]; // Z, C, O, Fe
    int FeedbackFlag;
    int TargetTask;
    int Index;
};

struct feedback_data_out {
    MyFloat dummy; // placeholder if needed for future return values
};

void apply_feedback_treewalk(double current_time, int feedback_type);

#endif
