import pandas as pd
import json
import random
from datetime import datetime, timedelta
import numpy as np

# Predefined options
skills_options = ["animal handling", "transport", "medical care", "foster care", "dog walking", "cat feeding", "admin work", "event setup", "vet tech", "Spanish-speaking"]
languages_options = ["en", "es", "pt", "fr"]
urgency_options = ["low", "medium", "high"]
experience_levels = ["beginner", "intermediate", "expert"]
preferred_animals = ["cats", "dogs", "kittens", "puppies", "birds"]
preferred_times = ["morning", "afternoon", "evening", "weekends"]

# Base location for Humboldt County, CA (Eureka)
base_lat = 40.8021
base_lon = -124.1637
lat_range = 0.1  # ~11 km
lon_range = 0.1

# Days of week
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Generate Tasks dataset (1000 entries)
task_data = []
for i in range(1000):
    task_id = f"T{i+1:03d}"
    num_skills = random.randint(1, 3)
    required_skills = random.sample(skills_options, num_skills)
    
    # Description
    desc_base = random.choice(["Transport ", "Feed ", "Walk ", "Medical check for ", "Foster care for ", "Admin for "]) + random.choice(["injured kitten", "hungry dog", "playful cat", "sick puppy", "adoption event", "supply distribution"])
    description = desc_base + ", requires gentle handling" if random.random() > 0.5 else desc_base
    
    # Schedule: Random future date in next 30 days, random time window
    start_date = datetime.now() + timedelta(days=random.randint(1, 30))
    start_hour = random.randint(8, 18)
    start_min = random.choice([0, 30])
    start_time = start_date.replace(hour=start_hour, minute=start_min, second=0, microsecond=0)
    duration = random.randint(30, 120)
    end_time = start_time + timedelta(minutes=duration)
    schedule = {
        "start": start_time.isoformat() + "Z",
        "end": end_time.isoformat() + "Z"
    }
    
    # Location: Random around base
    lat = base_lat + random.uniform(-lat_range, lat_range)
    lon = base_lon + random.uniform(-lon_range, lon_range)
    location = {"lat": round(lat, 4), "lon": round(lon, 4)}
    
    urgency = random.choice(urgency_options)
    num_lang = random.randint(0, 1)  # 0 or 1 language
    language = random.sample(languages_options, num_lang) if num_lang > 0 else []
    duration_minutes = duration
    
    task_data.append({
        "task_id": task_id,
        "required_skills": json.dumps(required_skills),
        "description": description,
        "schedule": json.dumps(schedule),
        "location": json.dumps(location),
        "urgency": urgency,
        "language": json.dumps(language),
        "duration_minutes": duration_minutes
    })

df_tasks = pd.DataFrame(task_data)
df_tasks.to_excel("tasks_dataset.xlsx", index=False)

# Generate Volunteers dataset (500 entries for balance)
volunteer_data = []
for i in range(500):
    vol_id = f"V{i+1:03d}"
    num_skills = random.randint(2, 5)
    skills = random.sample(skills_options, num_skills)
    
    # Availability: Random days and times
    avail = {}
    num_days = random.randint(1, 4)
    selected_days = random.sample(days, num_days)
    for day in selected_days:
        num_slots = random.randint(1, 2)
        slots = []
        for _ in range(num_slots):
            start_h = random.randint(8, 18)
            slots.append(f"{start_h:02d}:00-{start_h+3:02d}:00")
        avail[day] = slots
    
    # Location
    lat = base_lat + random.uniform(-lat_range, lat_range)
    lon = base_lon + random.uniform(-lon_range, lon_range)
    location = {"lat": round(lat, 4), "lon": round(lon, 4)}
    
    # Experience: For a few skills
    exp_skills = random.sample(skills_options, random.randint(1, 3))
    experience = {skill: random.choice(experience_levels) for skill in exp_skills}
    
    num_lang = random.randint(1, 2)
    languages = random.sample(languages_options, num_lang)
    
    avg_feedback = round(random.uniform(1.0, 5.0), 1)
    
    # Preferences
    prefs = {
        "preferred_animals": random.sample(preferred_animals, random.randint(1, 2)),
        "preferred_time": random.choice(preferred_times)
    }
    
    volunteer_data.append({
        "volunteer_id": vol_id,
        "skills": json.dumps(skills),
        "availability": json.dumps(avail),
        "location": json.dumps(location),
        "experience": json.dumps(experience),
        "languages": json.dumps(languages),
        "avg_feedback": avg_feedback,
        "preferences": json.dumps(prefs)
    })

df_volunteers = pd.DataFrame(volunteer_data)
df_volunteers.to_excel("volunteers_dataset.xlsx", index=False)

print("Datasets generated: tasks_dataset.xlsx (1000 rows) and volunteers_dataset.xlsx (500 rows)")