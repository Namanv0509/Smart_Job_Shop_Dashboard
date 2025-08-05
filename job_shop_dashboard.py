import streamlit as st
import collections
from ortools.sat.python import cp_model
import plotly.figure_factory as ff
import pandas as pd

st.title("Simplified Job Shop Scheduler")
st.write("Each job follows a fixed route across machines. Schedule shows only job-level summary.")

num_machines = st.number_input("Number of Machines", min_value=1, value=3)
num_jobs = st.number_input("Number of Jobs", min_value=1, value=3)

jobs_data = []
due_dates = []

st.subheader("Define Route and Processing Time for Each Job")
for job_id in range(num_jobs):
    st.write(f"### Job {job_id}")
    
    route = st.text_input(f"Machine Route for Job {job_id} (comma separated machine IDs)", value="0,1,2", key=f"route_{job_id}")
    durations = st.text_input(f"Processing Times for each machine (same length)", value="3,3,3", key=f"dur_{job_id}")
    
    try:
        machines = list(map(int, route.strip().split(',')))
        times = list(map(int, durations.strip().split(',')))
        if len(machines) != len(times):
            st.error(f"Job {job_id}: Mismatch in number of machines and durations")
        else:
            jobs_data.append(list(zip(machines, times)))
    except:
        st.error(f"Job {job_id}: Invalid format. Use comma-separated numbers.")
    
    due_date = st.number_input(f"Due Date for Job {job_id}", min_value=1, value=15, key=f"due_{job_id}")
    due_dates.append(due_date)

if st.button("Compute Schedule"):
    model = cp_model.CpModel()
    horizon = sum(t for job in jobs_data for _, t in job)
    all_machines = range(num_machines)

    task_type = collections.namedtuple("task_type", "start end interval")
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, (machine, duration) in enumerate(job):
            suffix = f'_{job_id}_{task_id}'
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Disjunctive: machines can’t process two jobs at once
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])
    
    # Precedence: enforce machine route per job
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)

    # Completion time = end of last task of each job
    job_ends = []
    for job_id, job in enumerate(jobs_data):
        last = len(job) - 1
        job_ends.append(all_tasks[job_id, last].end)

    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, job_ends)
    model.Minimize(obj_var)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        st.success("Schedule computed successfully!")

        summary = []
        gantt_data = []
        route_sheet_data = []
        machine_schedules = collections.defaultdict(list)

        for job_id, job in enumerate(jobs_data):
            last_task_id = len(job) - 1
            end_time = solver.Value(all_tasks[job_id, last_task_id].end)
            summary.append({
                "Job ID": job_id,
                "Completion Time": end_time,
                "Due Date": due_dates[job_id],
                "Delta": end_time - due_dates[job_id]
            })

            for task_id, (machine, duration) in enumerate(job):
                start = solver.Value(all_tasks[job_id, task_id].start)
                end = solver.Value(all_tasks[job_id, task_id].end)
                
                # For Gantt Chart
                gantt_data.append(dict(
                    Task=f"Machine {machine}",
                    Start=start,
                    Finish=end,
                    Resource=f"Job {job_id}"
                ))

                # For Route Sheet Table
                route_sheet_data.append({
                    "Job ID": job_id,
                    "Machine": machine,
                    "Start Time": start,
                    "End Time": end
                })

                # For Machine-wise Schedule
                machine_schedules[machine].append((start, job_id + 1, end))  # +1 for 1-based job indexing

        # Display Route Sheet Table
        st.subheader("🗺️ Final Route Sheet")
        st.dataframe(pd.DataFrame(route_sheet_data).sort_values(by=["Job ID", "Start Time"]))

        # Summary table
        st.subheader("📋 Job Summary Table")
        st.dataframe(pd.DataFrame(summary))

        # Gantt chart
        df_gantt = pd.DataFrame(gantt_data)
        fig = ff.create_gantt(df_gantt, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True)
        fig.update_layout(title="Job Routing Timeline", xaxis_title="Time", yaxis_title="Machines")
        st.plotly_chart(fig)

        # Machine-wise Route Sheet in Requested Format
        st.subheader("🛠️ Machine-wise Route Sheet")
        route_sheet_text = "int routeSheet[3][9] = {\n"
        for machine in all_machines:
            # Sort tasks by start time for each machine
            tasks = sorted(machine_schedules[machine], key=lambda x: x[0])
            # Create array of job_id, start, end
            task_sequence = []
            for start, job_id, end in tasks:
                task_sequence.extend([job_id, start, end])
            # Pad with zeros if fewer than 3 tasks (to match 9 elements)
            while len(task_sequence) < 9:
                task_sequence.append(0)
            # Create comment with job and time range details
            comment = f" // Machine {machine + 1}: " + ", ".join(
                f"Job {job_id} ({start}-{end}s)" 
                for start, job_id, end in tasks
            ) if tasks else f" // Machine {machine + 1}: No tasks"
            # Format the line
            route_sheet_text += f"  {{{', '.join(map(str, task_sequence))}}}{comment}\n"
        route_sheet_text += "};"
        st.code(route_sheet_text, language="cpp")

    else:
        st.error("No feasible solution found.")