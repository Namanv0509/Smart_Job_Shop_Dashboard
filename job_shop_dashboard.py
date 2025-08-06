import streamlit as st
import collections
import plotly.figure_factory as ff
import pandas as pd
from ortools.sat.python import cp_model

st.title("Job Shop Scheduler with Algorithm Selection")
st.write("Each job follows a fixed route across machines. Select a scheduling algorithm and optimization objective in the sidebar.")

# Sidebar for settings
st.sidebar.header("Scheduler Settings")
algorithm = st.sidebar.selectbox(
    "Select Scheduling Algorithm",
    ["OR-Tools (CP-SAT)", "Shortest Processing Time (SPT)", "Longest Processing Time (LPT)", "Earliest Due Date (EDD)"]
)
optimization_objective = st.sidebar.selectbox(
    "Optimization Objective",
    ["Minimize Makespan", "Minimize Tardiness"]
)
optimize_due_dates = optimization_objective == "Minimize Tardiness"

num_machines = st.number_input("Number of Machines", min_value=1, value=3)
num_jobs = st.number_input("Number of Jobs", min_value=1, value=3)

jobs_data = []
due_dates = []

st.subheader("Define Route, Processing Time, and Due Date for Each Job")
for job_id in range(num_jobs):
    st.write(f"### Job {job_id}")
    
    route = st.text_input(f"Machine Route for Job {job_id} (comma separated machine IDs)", value="0,1,2", key=f"route_{job_id}")
    durations = st.text_input(f"Processing Times for each machine (same length)", value="3,3,3", key=f"dur_{job_id}")
    
    try:
        machines = list(map(int, route.strip().split(',')))
        times = list(map(int, durations.strip().split(',')))
        if len(machines) != len(times):
            st.error(f"Job {job_id}: Mismatch in number of machines and durations")
            jobs_data.append(None)
        else:
            jobs_data.append(list(zip(machines, times)))
    except:
        st.error(f"Job {job_id}: Invalid format. Use comma-separated numbers.")
        jobs_data.append(None)
    
    due_date = st.number_input(f"Due Date for Job {job_id}", min_value=1, value=15, key=f"due_{job_id}")
    due_dates.append(due_date)

if st.button("Compute Schedule"):
    # Validate input data
    if any(job is None for job in jobs_data):
        st.error("Please correct all job inputs before computing the schedule.")
        st.stop()

    def compute_or_tools_schedule():
        model = cp_model.CpModel()
        horizon = sum(t for job in jobs_data for _, t in job)
        all_machines = range(num_machines)

        task_type = collections.namedtuple("task_type", "start end interval")
        all_tasks = {}
        machine_to_intervals = collections.defaultdict(list)
        tardiness_vars = []

        for job_id, job in enumerate(jobs_data):
            for task_id, (machine, duration) in enumerate(job):
                suffix = f'_{job_id}_{task_id}'
                start_var = model.NewIntVar(0, horizon, 'start' + suffix)
                end_var = model.NewIntVar(0, horizon, 'end' + suffix)
                interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval' + suffix)
                all_tasks[job_id, task_id] = task_type(start=start_var, end=end_var, interval=interval_var)
                machine_to_intervals[machine].append(interval_var)
            
            if optimize_due_dates:
                last_task_id = len(job) - 1
                tardiness = model.NewIntVar(0, horizon, f'tardiness_{job_id}')
                model.Add(tardiness >= all_tasks[job_id, last_task_id].end - due_dates[job_id])
                tardiness_vars.append(tardiness)

        for machine in all_machines:
            model.AddNoOverlap(machine_to_intervals[machine])
        
        for job_id, job in enumerate(jobs_data):
            for task_id in range(len(job) - 1):
                model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)

        if optimize_due_dates:
            total_tardiness = model.NewIntVar(0, horizon * num_jobs, 'total_tardiness')
            model.Add(total_tardiness == sum(tardiness_vars))
            model.Minimize(total_tardiness)
        else:
            obj_var = model.NewIntVar(0, horizon, 'makespan')
            job_ends = [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)]
            model.AddMaxEquality(obj_var, job_ends)
            model.Minimize(obj_var)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        return solver, status, all_tasks

    def compute_dispatching_rule_schedule(key_func, reverse=False):
        schedule = []
        machine_free_time = [0] * num_machines
        job_task_index = [0] * num_jobs
        job_completion_times = [0] * num_jobs
        machine_schedules = collections.defaultdict(list)

        # Sort jobs by the key function and job ID for tie-breaking
        job_metrics = [(i, key_func(jobs_data[i], due_dates[i])) for i in range(num_jobs)]
        job_order = [i for i, _ in sorted(job_metrics, key=lambda x: (x[1], x[0]), reverse=reverse)]

        # Track available tasks (job_id, task_id) that can be scheduled
        available_tasks = []
        for job_id in range(num_jobs):
            if job_task_index[job_id] < len(jobs_data[job_id]):
                available_tasks.append((job_id, job_task_index[job_id]))

        while available_tasks:
            # Find the task that can start earliest
            earliest_start = float('inf')
            next_task = None
            for job_id, task_id in available_tasks:
                machine, duration = jobs_data[job_id][task_id]
                start_time = max(machine_free_time[machine], job_completion_times[job_id])
                if start_time < earliest_start:
                    earliest_start = start_time
                    next_task = (job_id, task_id)

            if next_task is None:
                break

            job_id, task_id = next_task
            machine, duration = jobs_data[job_id][task_id]
            start_time = max(machine_free_time[machine], job_completion_times[job_id])
            end_time = start_time + duration

            schedule.append((job_id, task_id, machine, start_time, end_time))
            machine_schedules[machine].append((start_time, job_id + 1, end_time))
            machine_free_time[machine] = end_time
            job_completion_times[job_id] = end_time
            job_task_index[job_id] += 1

            # Update available tasks
            available_tasks = [(j, job_task_index[j]) for j in range(num_jobs) if job_task_index[j] < len(jobs_data[j])]

        return schedule, machine_schedules, job_completion_times

    # Compute schedule based on selected algorithm
    if algorithm == "OR-Tools (CP-SAT)":
        solver, status, all_tasks = compute_or_tools_schedule()
        if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
            st.error("No feasible solution found.")
            st.stop()
        st.success("Schedule computed successfully!")
        # Build machine_schedules for OR-Tools
        machine_schedules = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, (machine, _) in enumerate(job):
                start = solver.Value(all_tasks[job_id, task_id].start)
                end = solver.Value(all_tasks[job_id, task_id].end)
                machine_schedules[machine].append((start, job_id + 1, end))
    else:
        if algorithm == "Shortest Processing Time (SPT)":
            key_func = lambda job, _: sum(t for _, t in job)
            schedule, machine_schedules, job_completion_times = compute_dispatching_rule_schedule(key_func, reverse=False)
        elif algorithm == "Longest Processing Time (LPT)":
            key_func = lambda job, _: sum(t for _, t in job)
            schedule, machine_schedules, job_completion_times = compute_dispatching_rule_schedule(key_func, reverse=True)
        elif algorithm == "Earliest Due Date (EDD)":
            key_func = lambda _, due_date: due_date
            schedule, machine_schedules, job_completion_times = compute_dispatching_rule_schedule(key_func, reverse=False)
        
        st.success("Schedule computed successfully!")

    # Generate output
    summary = []
    gantt_data = []
    route_sheet_data = []

    if algorithm == "OR-Tools (CP-SAT)":
        for job_id, job in enumerate(jobs_data):
            last_task_id = len(job) - 1
            end_time = solver.Value(all_tasks[job_id, last_task_id].end)
            tardiness = max(0, end_time - due_dates[job_id])
            status = "Early" if end_time < due_dates[job_id] else "On Time" if end_time == due_dates[job_id] else "Delayed"
            summary.append({
                "Job ID": job_id,
                "Completion Time": end_time,
                "Due Date": due_dates[job_id],
                "Tardiness": tardiness,
                "Status": status
            })

            for task_id, (machine, duration) in enumerate(job):
                start = solver.Value(all_tasks[job_id, task_id].start)
                end = solver.Value(all_tasks[job_id, task_id].end)
                gantt_data.append(dict(Task=f"Machine {machine}", Start=start, Finish=end, Resource=f"Job {job_id}"))
                route_sheet_data.append({"Job ID": job_id, "Machine": machine, "Start Time": start, "End Time": end})
    else:
        for job_id, job in enumerate(jobs_data):
            end_time = job_completion_times[job_id]
            tardiness = max(0, end_time - due_dates[job_id])
            status = "Early" if end_time < due_dates[job_id] else "On Time" if end_time == due_dates[job_id] else "Delayed"
            summary.append({
                "Job ID": job_id,
                "Completion Time": end_time,
                "Due Date": due_dates[job_id],
                "Tardiness": tardiness,
                "Status": status
            })

        for job_id, task_id, machine, start, end in schedule:
            gantt_data.append(dict(Task=f"Machine {machine}", Start=start, Finish=end, Resource=f"Job {job_id}"))
            route_sheet_data.append({"Job ID": job_id, "Machine": machine, "Start Time": start, "End Time": end})

    # Display outputs
    st.subheader("🗺️ Final Route Sheet")
    st.dataframe(pd.DataFrame(route_sheet_data).sort_values(by=["Job ID", "Start Time"]))

    st.subheader("📋 Job Summary Table")
    st.dataframe(pd.DataFrame(summary))

    # Gantt chart with custom colors
    df_gantt = pd.DataFrame(gantt_data)
    # Define colors: Job 0 = red, Job 1 = green, Job 2 = blue, others cycle through hex codes
    colors = {
        f"Job {i}": color for i, color in enumerate(
            ["#FF0000", "#00FF00", "#0000FF"] + ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Hex codes for red, green, blue, then defaults
        ) if i < num_jobs
    }
    fig = ff.create_gantt(df_gantt, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True, colors=colors)
    fig.update_layout(
        title="Job Routing Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Machines",
        xaxis=dict(
            tickvals=list(range(0, int(max(df_gantt['Finish']) + 1), 1)),  # Show ticks at every second
            ticktext=[str(i) for i in range(0, int(max(df_gantt['Finish']) + 1), 1)],  # Label ticks as seconds
            type='linear'  # Ensure x-axis is treated as numerical, not dates
        )
    )
    st.plotly_chart(fig)

    st.subheader("🛠️ Machine-wise Route Sheet")
    route_sheet_text = "int routeSheet[3][9] = {\n"
    for machine in range(num_machines):
        tasks = sorted(machine_schedules[machine], key=lambda x: x[0])
        task_sequence = []
        for start, job_id, end in tasks:
            task_sequence.extend([job_id, start, end])
        while len(task_sequence) < 9:
            task_sequence.append(0)
        comment = f" // Machine {machine + 1}: " + ", ".join(
            f"Job {job_id} ({start}-{end}s)" 
            for start, job_id, end in tasks
        ) if tasks else f" // Machine {machine + 1}: No tasks"
        route_sheet_text += f"  {{{', '.join(map(str, task_sequence))}}}{comment}\n"
    route_sheet_text += "};"
    st.code(route_sheet_text, language="cpp")