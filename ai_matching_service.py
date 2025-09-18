import pandas as pd
import json
import numpy as np
from scipy.spatial.distance import cosine
from datetime import datetime
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Load datasets with validation
def load_datasets(volunteers_file: str, tasks_file: str) -> tuple:
    df_volunteers = pd.read_excel(volunteers_file)
    df_tasks = pd.read_excel(tasks_file)
    
    # Parse JSON columns for volunteers
    json_cols_vol = ['skills', 'availability', 'location', 'experience', 'languages', 'preferences']
    for col in json_cols_vol:
        if col in df_volunteers.columns:
            df_volunteers[col] = df_volunteers[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip() else []
            )
    
    # Parse JSON columns for tasks
    json_cols_task = ['required_skills', 'schedule', 'location', 'language']
    for col in json_cols_task:
        if col in df_tasks.columns:
            df_tasks[col] = df_tasks[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip() else []
            )
    
    # Validate skills and required_skills
    df_volunteers = df_volunteers[df_volunteers['skills'].apply(
        lambda x: isinstance(x, list) and len(x) > 0 and all(isinstance(s, str) and s.strip() for s in x)
    )]
    df_tasks = df_tasks[df_tasks['required_skills'].apply(
        lambda x: isinstance(x, list) and len(x) > 0 and all(isinstance(s, str) and s.strip() for s in x)
    )]
    
    if df_volunteers.empty or df_tasks.empty:
        raise ValueError("No valid volunteers or tasks after filtering invalid skills.")
    
    return df_volunteers, df_tasks

# Simple TF-IDF for skill matching
def compute_tfidf(texts: List[str]) -> np.ndarray:
    # Handle empty texts
    texts = [t if t.strip() else "none" for t in texts]  # Replace empty strings with "none"
    
    all_tokens = set()
    doc_freq = {}
    tf_matrices = []
    for text in texts:
        tokens = text.lower().split()
        if not tokens:  # Handle empty tokens
            tokens = ["none"]
        all_tokens.update(tokens)
        doc_freq.update({token: doc_freq.get(token, 0) + 1 for token in set(tokens)})
        tf = {token: count / len(tokens) for token, count in dict.fromkeys(tokens, 1).items()}
        tf_matrices.append(tf)
    
    n_docs = len(texts)
    unique_tokens = list(all_tokens)
    tfidf = np.zeros((n_docs, len(unique_tokens)))
    for i, tf in enumerate(tf_matrices):
        for j, token in enumerate(unique_tokens):
            if token in tf:
                idf = np.log(n_docs / (1 + doc_freq.get(token, 0)))
                tfidf[i, j] = tf[token] * idf
    return tfidf

# Skill match score
def skill_match_score(vol_skills: List[str], task_skills: List[str]) -> float:
    if not vol_skills or not task_skills or not all(isinstance(s, str) and s.strip() for s in vol_skills + task_skills):
        return 0.0
    vol_text = ' '.join(vol_skills)
    task_text = ' '.join(task_skills)
    tfidf_vol = compute_tfidf([vol_text])[0]
    tfidf_task = compute_tfidf([task_text])[0]
    if len(tfidf_vol) != len(tfidf_task):
        min_len = min(len(tfidf_vol), len(tfidf_task))
        tfidf_vol = tfidf_vol[:min_len]
        tfidf_task = tfidf_task[:min_len]
    similarity = 1 - cosine(tfidf_vol, tfidf_task)
    return similarity if not np.isnan(similarity) else 0.0

# Availability check
def check_availability(vol_availability: Dict, task_schedule: Dict) -> bool:
    task_start = datetime.fromisoformat(task_schedule['start'].replace('Z', '+00:00'))
    task_day = task_start.strftime('%A')
    if task_day not in vol_availability:
        return False
    task_time = task_start.time()
    for slot in vol_availability[task_day]:
        slot_start, slot_end = slot.split('-')
        slot_start_time = datetime.strptime(slot_start, '%H:%M').time()
        slot_end_time = datetime.strptime(slot_end, '%H:%M').time()
        if slot_start_time <= task_time <= slot_end_time:
            return True
    return False

# Location distance
def calculate_distance(vol_loc: Dict, task_loc: Dict) -> float:
    return haversine(
        vol_loc['lat'], vol_loc['lon'],
        task_loc['lat'], task_loc['lon']
    )

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371  # km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Language match
def check_language_match(vol_languages: List[str], task_languages: List[str]) -> bool:
    return bool(set(task_languages) & set(vol_languages)) if task_languages else True

# Matching function
def match_volunteers_to_task(task: Dict, df_volunteers: pd.DataFrame, top_k: int = 5) -> List[Dict]:
    matches = []
    for _, vol_row in df_volunteers.iterrows():
        vol = vol_row.to_dict()
        
        # Skill score
        skill_score = skill_match_score(vol['skills'], task['required_skills'])
        
        # Availability score
        avail_score = 1.0 if check_availability(vol['availability'], task['schedule']) else 0.0
        
        # Location score (normalize to 50km max)
        dist = calculate_distance(vol['location'], task['location'])
        location_score = max(0.0, 1.0 - (dist / 50.0))
        
        # Language score
        lang_score = 1.0 if check_language_match(vol['languages'], task['language']) else 0.0
        
        # Experience boost (0-0.2)
        exp_boost = 0.0
        for skill in task['required_skills']:
            if skill in vol['experience'] and vol['experience'][skill] != 'beginner':
                exp_boost = 0.2
                break
        
        # Feedback score (normalized 0-1)
        feedback_score = vol['avg_feedback'] / 5.0
        
        # Urgency weighting (higher for urgent tasks)
        urgency_weights = {'low': 0.1, 'medium': 0.2, 'high': 0.3}
        urgency_factor = urgency_weights.get(task['urgency'], 0.2) * feedback_score
        
        # Total score (weighted sum)
        total_score = (
            0.4 * skill_score +
            0.2 * avail_score +
            0.15 * location_score +
            0.1 * lang_score +
            0.05 * exp_boost +
            0.1 * feedback_score +
            urgency_factor
        )
        
        if total_score > 0.3:  # Threshold for viable matches
            matches.append({
                'volunteer_id': vol['volunteer_id'],
                'total_score': round(total_score, 3),
                'skill_score': round(skill_score, 3),
                'avail_score': avail_score,
                'location_score': round(location_score, 3),
                'lang_score': lang_score,
                'exp_boost': exp_boost,
                'feedback_score': round(feedback_score, 3),
                'distance_km': round(dist, 2),
                'explanation': f"Strong skills match ({len(set(vol['skills']) & set(task['required_skills']))}/{len(task['required_skills'])}), {'available' if avail_score else 'not available'}, {round(dist, 1)}km away"
            })
    
    # Sort by total score descending
    matches.sort(key=lambda x: x['total_score'], reverse=True)
    return matches[:top_k]

# FastAPI app for microservice
app = FastAPI(title="AI Matching Service")

class TaskRequest(BaseModel):
    task_id: str

df_volunteers, df_tasks = load_datasets('volunteers_dataset.xlsx', 'tasks_dataset.xlsx')

@app.post("/recommend-volunteers")
async def recommend_volunteers(request: TaskRequest):
    task_row = df_tasks[df_tasks['task_id'] == request.task_id]
    if task_row.empty:
        raise HTTPException(status_code=404, detail="Task not found")
    task = task_row.iloc[0].to_dict()
    matches = match_volunteers_to_task(task, df_volunteers)
    return {"task_id": request.task_id, "matches": matches}

if __name__ == "__main__":
    # Standalone test example
    sample_task = df_tasks.iloc[0].to_dict()  # First task
    matches = match_volunteers_to_task(sample_task, df_volunteers)
    print("Sample Matches for Task", sample_task['task_id'])
    for m in matches:
        print(m)
    
    # Run API: uvicorn ai_matching_service:app --reload
    # uvicorn.run(app, host="0.0.0.0", port=8000)