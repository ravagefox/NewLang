// =====================================
// Job Runner with progress + state
// =====================================

// ---------- Data types ----------
declare
type struct {
    array name;        // array<char> (use std:toArray("..."))
    int   weight;      // relative weight for overall progress
    int   steps;       // how many inner steps to simulate
    ctor(name, weight, steps);
} Task;

extend type Task:ctor(name, weight, steps) {
    this->name   = name;
    this->weight = weight;
    this->steps  = steps;
}

// Snapshot of current progress (emitted to callbacks)
declare
type struct {
    int   taskIndex;       // 1-based
    int   taskCount;
    array taskName;        // array<char>
    number taskPercent;    // 0..100
    number overallPercent; // 0..100
    int   state;           // 0=Idle,1=Running,2=Done,3=Cancelled
    ctor(taskIndex, taskCount, taskName, taskPercent, overallPercent, state);
} Progress;

extend type Progress:ctor(i, n, nm, tp, op, st) {
    this->taskIndex       = i;
    this->taskCount       = n;
    this->taskName        = nm;
    this->taskPercent     = tp;
    this->overallPercent  = op;
    this->state           = st;
}

// The runner holds a pre-sized array of tasks and executes them.
declare
type struct {
    array tasks;       // storage
    int   count;       // how many used
    int   capacity;
    int   totalWeight;
    int   doneWeight;
    int   state;       // 0=Idle,1=Running,2=Done,3=Cancelled
    int   cancelFlag;  // 0/1
    ctor(capacity);
} TaskRunner;

extend type TaskRunner:ctor(capacity) {
    this->tasks       = new[capacity];
    this->count       = 0;
    this->capacity    = capacity;
    this->totalWeight = 0;
    this->doneWeight  = 0;
    this->state       = 0;  // Idle
    this->cancelFlag  = 0;
}

extend type TaskRunner:addTask(t) {
    // naive push into fixed array
    auto c = this->count;
    auto ts = this->tasks;
    ts[c] = t;
    this->count = c + 1;
    this->totalWeight = this->totalWeight + t->weight;
    this->tasks = ts;
}

extend type TaskRunner:cancel() {
    this->cancelFlag = 1;
}

// ---------- Utility “work” ----------
// Burn CPU cycles; return some number to keep math “live”.
def spin(n) {
    auto acc = 0;
    auto i = 0;
    for (i = 0; i < n; i = i + 1) {
        // simple floating-ish math to keep Python from optimizing too much
        acc = acc + (i * 1.0) / (i + 1);
    }
    finalize acc;
}

// ---------- Runner execution ----------
// progress: delegate collection (new[](&)())
// workFactor: tune this so total runtime ≈ 5 minutes on your machine
extend type TaskRunner:run(progress, workFactor) -> int {
    if this->count == 0 {
        std:println("No tasks.\n");
        finalize 0;
    }

    this->state = 1; // Running

    auto idx = 0;
    for (idx = 0; idx < this->count; idx = idx + 1) {
        if this->cancelFlag == 1 {
            this->state = 3; // Cancelled
            // emit final snapshot
            Progress snap(idx, this->count, std:toArray("cancelled"), 0, (this->doneWeight * 100.0) / this->totalWeight, this->state);
            progress->call(snap);
            finalize 0;
        }

        auto t = this->tasks[idx];
        auto steps = t->steps;
        auto step = 0;

        // per-task progress threshold (emit every 10%)
        auto nextPct = 10.0;

        for (step = 0; step < steps; step = step + 1) {
            // simulate unit of work
            spin(workFactor);

            // compute task and overall percentages
            auto taskPct = (step * 100.0) / steps;
            auto overall = ( (this->doneWeight * 100.0)
                           + (t->weight * taskPct) ) / this->totalWeight;

            // emit progress at 0%,10%,20%,...,100% (and always at last step)
            if (taskPct >= nextPct or step == steps - 1) {
                Progress snap(idx + 1, this->count, t->name, taskPct, overall, this->state);
                
                progress->call(snap);
                nextPct = nextPct + 10.0;
            }

            if this->cancelFlag == 1 {
                this->state = 3; // Cancelled
                auto snap2 = Progress(idx + 1, this->count, t->name, taskPct, overall, this->state);
                progress->call(snap2);
                finalize -1;
            }
        }

        // mark task done
        this->doneWeight = this->doneWeight + t->weight;
    }

    this->state = 2; // Done
    // final overall 100% snapshot
    Progress lastSnap(this->count, this->count, std:toArray("done"), 100.0, 100.0, this->state);
    progress->call(lastSnap);

    finalize 0;
}

// =====================================
// Demo main
// =====================================
__main__
def foo() : void {
    // Runner with room for 3 tasks
    TaskRunner runner(3);
    Task t1(std:toArray("Warm-up"),     1, 50);
    Task t2(std:toArray("Crunching"),     3, 120);
    Task t3(std:toArray("Finalizing"),     2, 80);

    // Add tasks (name, weight, steps). More steps = more granularity.
    runner->addTask(t1);

    // Progress callback: pretty-print updates
    auto onProgress = new[](&)();
    onProgress->add([args](&)(def {
        auto p = args[1];  // Progress object
        
        std:println("Task ");        std:println(p->taskIndex);
        std:println("/");            std:println(p->taskCount);
        std:println("  name: ");     std:println(p->taskName);
        std:println("  task%: ");    std:println(p->taskPercent);
        std:println("  overall%: "); std:println(p->overallPercent);
        std:println("  state: ");    std:println(p->state);

    }));

    // Tune workFactor to roughly 5 minutes on your machine.
    // Start with something like 400000 for a quick test, then increase.
    auto workFactor = 4; // <-- increase this to slow down / ~5min total

    std:println("Starting runner...\n");
    runner->run(onProgress, workFactor);
    std:println("All done.\n");
}
