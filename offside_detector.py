import cv2
import numpy as np
from collections import defaultdict

class TeamClassifier:
    def __init__(self):
        self.team_colors = {}
    
    def classify_players(self, detections, frame):
        if len(self.team_colors) < 2 and len(detections) >= 2:
            for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']].values):
                if cls in ['player', 'goalkeeper']:
                    jersey_roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    if jersey_roi.size > 0:
                        dominant_color = self.get_dominant_color(jersey_roi)
                        if not self.team_colors:
                            self.team_colors[0] = dominant_color
                        else:
                            if all(np.linalg.norm(color - dominant_color) > 50 for color in self.team_colors.values()):
                                self.team_colors[1] = dominant_color

        team_assignments = {}
        player_positions = []
        for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']].values):
            if cls in ['player', 'goalkeeper']:
                player_positions.append((x1 + x2) / 2)
                if self.team_colors:
                    jersey_roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    if jersey_roi.size > 0:
                        dominant_color = self.get_dominant_color(jersey_roi)
                        distances = [np.linalg.norm(dominant_color - color) for color in self.team_colors.values()]
                        best_team = np.argmin(distances)
                        if distances[best_team] < 75:
                            team_assignments[i] = best_team

        if len(self.team_colors) < 2 and len(player_positions) >= 2:
            median_x = np.median(player_positions)
            for i, pos in enumerate(player_positions):
                team_assignments[i] = 0 if pos < median_x else 1

        return team_assignments

    def get_dominant_color(self, roi):
        pixels = roi.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return centers[0]

class OffsideDetector:
    def __init__(self):
        self.team_classifier = TeamClassifier()
        self.playing_direction = 1

    def infer_playing_direction(self, teams, frame_width):
        if len(teams) < 2:
            return self.playing_direction

        team_avg_positions = {}
        for t, players in teams.items():
            positions = [(p['xmin'] + p['xmax']) / 2 for p in players]
            if positions:
                team_avg_positions[t] = np.mean(positions)

        if len(team_avg_positions) < 2:
            return self.playing_direction

        t0, t1 = team_avg_positions[0], team_avg_positions[1]
        self.playing_direction = 1 if t0 < t1 else -1
        return self.playing_direction

    def detect_offside(self, detections, frame):
        team_assignments = self.team_classifier.classify_players(detections, frame)

        teams = defaultdict(list)
        for i, row in detections.iterrows():
            if row['class_name'] == 'player':
                team = team_assignments.get(i, i % 2)
                teams[team].append(row)

        self.playing_direction = self.infer_playing_direction(teams, frame.shape[1])

        ball_detections = detections[detections['class_name'] == 'ball']
        ball_visible = not ball_detections.empty
        ball_x = (ball_detections.iloc[0]['xmin'] + ball_detections.iloc[0]['xmax']) / 2 if ball_visible else None

        if ball_visible:
            self.last_ball_x = ball_x

        attacking_team = 0 if self.playing_direction == 1 else 1
        defending_team = 1 - attacking_team

        defenders = sorted(teams[defending_team], key=lambda p: ((p['xmin'] + p['xmax']) / 2) * self.playing_direction)

        if len(defenders) >= 2:
            second_last_defender = defenders[-2]
            offside_line_x = (second_last_defender['xmin'] + second_last_defender['xmax']) / 2
        elif defenders:
            offside_line_x = (defenders[0]['xmin'] + defenders[0]['xmax']) / 2
        else:
            offside_line_x = frame.shape[1] / 2

        attacker_positions = [(p['xmin'] + p['xmax']) / 2 for p in teams[attacking_team]]
        defender_positions = [(p['xmin'] + p['xmax']) / 2 for p in teams[defending_team]]

        offside_players = []
        for p in teams[attacking_team]:
            px = (p['xmin'] + p['xmax']) / 2
            in_opponent_half = (px > frame.shape[1] / 2 if self.playing_direction == 1 else px < frame.shape[1] / 2)
            if not in_opponent_half:
                continue

            ahead_of_defenders = (px - offside_line_x) * self.playing_direction > 0
            ahead_of_ball = True if not ball_visible else (px - ball_x) * self.playing_direction > 0

            if ahead_of_defenders and ahead_of_ball:
                offside_players.append(p)

        if not offside_players and len(attacker_positions) >= 2 and len(defender_positions) >= 2:
            attacker_sorted = sorted(attacker_positions, reverse=self.playing_direction == -1)
            defender_sorted = sorted(defender_positions, reverse=self.playing_direction == -1)
            one_behind = (attacker_sorted[-1] - defender_sorted[-1]) * self.playing_direction < 0
            one_ahead = (attacker_sorted[0] - defender_sorted[0]) * self.playing_direction > 0
            if one_behind and one_ahead:
                offside_players = teams[attacking_team]

        return offside_players

def draw_definitive_results(frame, detections, offside_players, detector):
    frame = frame.copy()

    players = detections[detections['class_name'].isin(['player', 'goalkeeper'])]
    if len(players) >= 2:
        defending_team = 1 if detector.playing_direction == 1 else 0
        defenders = [p for _, p in players.iterrows()
                     if (p['xmin'] + p['xmax']) / 2 * detector.playing_direction < frame.shape[1] / 2]

        if len(defenders) >= 2:
            defenders_sorted = sorted(defenders, key=lambda x: (x['xmin'] + x['xmax']) / 2 * detector.playing_direction)
            offside_line_x = int((defenders_sorted[-2]['xmin'] + defenders_sorted[-2]['xmax']) / 2)
            cv2.line(frame, (offside_line_x, 0), (offside_line_x, frame.shape[0]), (0, 255, 255), 2)

    for _, player in players.iterrows():
        is_offside = any((p['xmin'] == player['xmin'] and p['ymin'] == player['ymin']) for p in offside_players)
        x1, y1, x2, y2 = int(player['xmin']), int(player['ymin']), int(player['xmax']), int(player['ymax'])
        
        if player['class_name'] == 'goalkeeper':
            color = (255, 0, 0)  # Blue for goalkeepers
            label = "GK"
        else:
            color = (0, 0, 255) if is_offside else (0, 255, 0)
            label = "OFF" if is_offside else "DEF"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    balls = detections[detections['class'] == 'ball']
    if len(balls) > 0:
        ball = balls.iloc[0]
        x1, y1, x2, y2 = int(ball['xmin']), int(ball['ymin']), int(ball['xmax']), int(ball['ymax'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    status = "OFFSIDE" if offside_players else "NO OFFSIDE"
    color = (0, 0, 255) if offside_players else (0, 255, 0)
    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame
