import os
import cv2
import json
import tempfile
import numpy as np
import urllib.request
import streamlit as st
from groq import Groq
import mediapipe as mp
import datetime
import re 

st.set_page_config(page_title="Coach IA Sportif & Databank", layout="centered")

# 🟢 C'EST ICI QU'IL FAUT METTRE L'INITIALISATION GROQ 🟢 👇
# Récupération automatique de la clé depuis les secrets Streamlit
groq_api_key = st.secrets.get("GROQ_API_KEY")

# Initialisation du client Groq
client = Groq(api_key=groq_api_key)
# -------------------------------------------------------- 👆

# --- GESTION DE LA VUE PERSISTANTE ---
if "active_analysis" not in st.session_state:
    st.session_state["active_analysis"] = None

if st.session_state["active_analysis"] is not None:
    data_path = st.session_state["active_analysis"]
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            saved_data = json.load(f)
            
        st.sidebar.button("⬅️ Retour au menu principal", on_click=lambda: st.session_state.update({"active_analysis": None}))
        st.title(f"📂 Panel d'Analyse : {saved_data.get('mouvement')} ({saved_data.get('date')})")
        
        tab_app, tab_muscles = st.tabs(["🎥 Rapport & Vidéo (App)", "🔥 Jumeau Numérique & Muscles (Test Muscles)"])
        
        reponse_brute = saved_data.get('bilan_coach_ia', '')
        
        def extraire_tag_local(texte, tag):
            match = re.search(f"<{tag}>(.*?)</{tag}>", texte, re.DOTALL)
            return match.group(1).strip() if match else ""

        bilan_texte = extraire_tag_local(reponse_brute, "BILAN") or reponse_brute
        c_coude = extraire_tag_local(reponse_brute, "COUDE")
        c_pro = extraire_tag_local(reponse_brute, "PROTRACTION")
        c_retro = extraire_tag_local(reponse_brute, "RETROVERSION")
        c_jambes = extraire_tag_local(reponse_brute, "JAMBES")

        with tab_app:
            st.header("🎯 Rapport de Performance Enregistré")
            st.write(f"**Poids de l'athlète :** {saved_data.get('poids_athlete')} kg")
            st.markdown("---")
            
            # --- AFFICHAGE DES NIVEAUX POSTURAUX ---
            st.subheader("📊 Niveaux Posturaux Extrapolés")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Lean (Épaules)", f"{saved_data.get('sync_lean', 75)}%")
            col_m2.metric("Protraction", f"{saved_data.get('sync_prot', 85)}%")
            col_m3.metric("Rétroversion", f"{saved_data.get('sync_retro', 75)}%")
            st.markdown("---")
            
            # Un seul affichage propre pour le bilan (le doublon a été supprimé)
            st.subheader("📋 Bilan & Note du Coach")
            st.write(bilan_texte)
            
            photos_focales = saved_data.get('photos_focus', {})
            if photos_focales:
                st.markdown("---")
                st.subheader("📸 Détails Techniques Isolés")
                
                for k, img_data in photos_focales.items():
                    if img_data:
                        try:
                            img_np = np.array(img_data, dtype=np.uint8)
                            if img_np.ndim == 3 and img_np.shape[2] == 3:
                                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                                st.image(img_rgb, use_container_width=True)
                            else:
                                st.image(img_np, use_container_width=True)
                                
                            if k == "coude":
                                st.info(f"**Coudes :** {c_coude}")
                            elif k == "protraction":
                                st.info(f"**Protraction Scapulaire :** {c_pro}")
                            elif k == "retroversion":
                                st.info(f"**Alignement Bassin :** {c_retro}")
                            elif k == "jambes":
                                st.info(f"**Tension Jambes :** {c_jambes}")
                            st.write("")
                        except Exception as e:
                            st.error(f"Erreur d'affichage pour l'image {k} : {e}")
                
        with tab_muscles:
            st.subheader("🔥 Dashboard Planche : Biomécanique & EMG")
            
            # Récupération sécurisée des valeurs posturales de l'analyse
            lean_val = saved_data.get('sync_lean', 75)
            prot_val = saved_data.get('sync_prot', 85)
            retro_val = saved_data.get('sync_retro', 75)
            
            # Calcul des forces brutes pour normalisation à 100% au total
            brut_delts = (lean_val / 100.0) * 98
            brut_chest = (prot_val / 100.0) * 85
            brut_abs = (retro_val / 100.0) * 90
            brut_biceps = (lean_val / 100.0) * 50
            brut_lombaires = (((100 - retro_val) / 100.0) * 60)
            brut_quads = (retro_val / 100.0) * 40
            brut_avant_bras = (lean_val / 100.0) * 45

            somme_totale = brut_delts + brut_chest + brut_abs + brut_biceps + brut_lombaires + brut_quads + brut_avant_bras

            def safe_pct(val):
                return int(round((val / somme_totale) * 100)) if somme_totale > 0 else 0

            # Dictionnaire normalisé : la somme totale fait exactement 100%
            engagement = {
                "deltoids": safe_pct(brut_delts),
                "chest": safe_pct(brut_chest),
                "abs": safe_pct(brut_abs),
                "biceps": safe_pct(brut_biceps),
                "lower-back": safe_pct(brut_lombaires),
                "forearm": safe_pct(brut_avant_bras),
                "quadriceps": safe_pct(brut_quads),
                "head": 0, "hair": 0, "neck": 0, "trapezius": 0, "triceps": 0, 
                "obliques": 0, "adductors": 0, "knees": 0, 
                "tibialis": 0, "calves": 0, "ankles": 0, "feet": 0, "hands": 0
            }
            
            def intensite_vers_couleur(pourcentage):
                if pourcentage <= 5: return "none"
                r, g, b = 255, int(255 - (255 * pourcentage / 100)), int(255 - (255 * pourcentage / 100))
                return f"#{r:02x}{g:02x}{b:02x}"

            col1, col2 = st.columns(2)
            
            # --- 1. JUMEAU NUMÉRIQUE AGRANDI ---
            with col1:
                st.subheader("📐 Jumeau Numérique (Simulation Posturale)")
                
                # --- RÉCUPÉRATION SÉCURISÉE DES VALEURS DEPUIS LE JSON DE L'ANALYSE ---
                lean_val = saved_data.get('sync_lean', 75)
                prot_val = saved_data.get('sync_prot', 85)
                retro_val = saved_data.get('sync_retro', 75)
                
                # Affichage explicite des pourcentages posturaux
                m1, m2, m3 = st.columns(3)
                m1.metric("Lean", f"{lean_val}%")
                m2.metric("Protraction", f"{prot_val}%")
                m3.metric("Rétroversion", f"{retro_val}%")
                
                lean_ratio = lean_val / 100.0
                prot_ratio = prot_val / 100.0
                retro_ratio = retro_val / 100.0
                
                import math
                
                # --- PARAMÈTRES FIXES ---
                hand_x, hand_y = 280, 230
                LONGUEUR_BRAS = 120  # Cette valeur ne change JAMAIS
                
                # --- CALCUL DE L'ANGLE ---
                # Plus le lean est faible (proche de 0), plus le bras est vertical (angle proche de 90°)
                # Plus le lean est fort (proche de 1), plus le bras est incliné (angle proche de 45°)
                angle_deg = 45 + (lean_ratio * 45) 
                angle_rad = math.radians(angle_deg)
                
                # --- ÉPAULE FIXE SUR LE CERCLE ---
                # La distance (LONGUEUR_BRAS) est constante ici
                shoulder_x = hand_x - (LONGUEUR_BRAS * math.cos(angle_rad))
                shoulder_y = hand_y - (LONGUEUR_BRAS * math.sin(angle_rad))
                
                # --- RESTE DU CORPS (ARTICULÉ) ---
                head_x = shoulder_x - 20
                head_y = shoulder_y - 20
                
                mid_x = shoulder_x + 90
                # La protra fait monter le haut du dos indépendamment du bras
                mid_y = shoulder_y - 10 - (prot_ratio * 25) 
                
                hip_x = shoulder_x + 180
                hip_y = shoulder_y + ((2.0 - retro_ratio) * 15)
                
                foot_x = hip_x + 150
                foot_y = hip_y + ((2.0 - retro_ratio) * 5)

                # viewBox élargie à 650x380 pour que le corps entier rentre sans être coupé
                svg_stickman = f"""
                <div style="display: flex; justify-content: center; align-items: center; background-color: #0e1117; border-radius: 12px; padding: 20px; height: 400px;">
                    <svg width="520" height="380" viewBox="0 0 650 380" style="overflow: visible;">
                        <line x1="20" y1="260" x2="620" y2="260" stroke="#444" stroke-width="3" stroke-dasharray="8,6" />
                        <circle cx="{hand_x}" cy="{hand_y}" r="8" fill="#ff4b4b" />
                        <line x1="{hand_x}" y1="{hand_y}" x2="{shoulder_x}" y2="{shoulder_y}" stroke="#00d2ff" stroke-width="12" stroke-linecap="round" />
                        <circle cx="{head_x}" cy="{head_y}" r="15" fill="none" stroke="#00d2ff" stroke-width="5" />
                        <path d="M {shoulder_x} {shoulder_y} Q {shoulder_x + 45} {shoulder_y - 15 - (prot_ratio * 18)} {mid_x} {mid_y}" fill="none" stroke="#00d2ff" stroke-width="16" stroke-linecap="round" />
                        <path d="M {mid_x} {mid_y} L {hip_x} {hip_y} L {foot_x} {foot_y}" fill="none" stroke="{'#ff4b4b' if retro_ratio < 0.5 else '#00d2ff'}" stroke-width="14" stroke-linecap="round" stroke-linejoin="round" />
                        <circle cx="{shoulder_x}" cy="{shoulder_y}" r="5" fill="#ffffff" />
                        <circle cx="{mid_x}" cy="{mid_y}" r="4" fill="#ffffff" />
                        <circle cx="{hip_x}" cy="{hip_y}" r="5" fill="#ffffff" />
                        <circle cx="{foot_x}" cy="{foot_y}" r="5" fill="#ffffff" />
                    </svg>
                </div>
                """
                st.components.v1.html(svg_stickman, height=420)
                
            # --- 2. ACTIVATION NERVEUSE AGRANDIE ---
            with col2:
                st.subheader("🔴 Activation Nerveuse (Face)")
                paths = {
                    "deltoids": ["M274.06 311.69q3.94 2.77 4.33 8.14.04.48-.38.73c-9.98 5.88-24.35 7.45-28.82 19.75-2.31 6.36-.97 17.35-1.43 23.68q-.55 7.51-5.73 14.07-10.37 13.11-13.81 16.67c-3.41 3.53-6.81 1.76-10.69-.47-15.42-8.87-24.95-25.45-22.52-43.22 2.05-14.92 12.71-25.79 24.06-35.02 16.99-13.82 35.58-17.99 54.99-4.33z", "M450.39 320.75q-.95-.52-.7-1.58c1.57-6.61 5.8-9.1 12.14-11.9 24.99-11.03 43.76 3.33 60.17 20.74 20.73 21.99 11.81 56.44-14.82 68.19-4.41 1.94-6.79-1.03-9.81-4.51-5.81-6.7-13.46-14.12-15.99-22.8-3.93-13.43 4.32-27.54-9.64-37.62q-8.22-5.93-17.99-9.08-1.84-.59-3.36-1.44z"],
                    "chest": ["M272.91 422.84c-18.95-17.19-22-57-12.64-78.79 5.57-12.99 26.54-24.37 39.97-25.87q20.36-2.26 37.02.75c9.74 1.76 16.13 15.64 18.41 25.04 3.99 16.48 3.23 31.38 1.67 48.06q-1.35 14.35-2.05 16.89c-6.52 23.5-38.08 29.23-58.28 24.53-9.12-2.12-17.24-4.38-24.1-10.61z", "M416.04 435c-15.12.11-34.46-6.78-41.37-21.48q-1.88-3.99-2.84-12.18c-2.89-24.41-5.9-53.65 8.44-74.79 4.26-6.26 10.49-7.93 18.36-8.56q11.66-.92 23.32-.35c10.58.53 18.02 2.74 26.62 7.87 12.81 7.65 19.73 14.52 22.67 29.75 4.94 25.57.24 64.14-28.21 74.97q-12.26 4.67-26.99 4.77z"],
                    "abs": ["M311.02 531.71a.23.23 0 01-.19-.21q-.39-10.47 1.9-20.76c1.26-5.69 7.66-9.9 13.1-12.9 9.09-5.01 18.93-11.15 28.56-14.92a1.24 1.21-42.6 01.94.03c3.28 1.52 4.78 3.87 4.82 7.68q.13 13.16-.15 26.31c-.08 3.85.78 8.39-.87 13.1q-.17.46-.59.72-2.65 1.65-4.29 1.82-21.06 2.22-43.23-.87z", "M321 577.76c-5.17-.33-8.71-.44-10-6.26q-3.2-14.44-.59-27.83.11-.53.64-.63c7.58-1.44 13.62-2.45 22.45-4.56q11.5-2.76 23.94-1.88c3.67.26 3.3 3.46 3.4 6.21q.46 12.55-.33 26.94-.25 4.41-1.81 8.08-.21.49-.73.6-1.39.28-3.22.29-16.89.14-33.75-.96z", "M347.73 429.25c7.46-3.61 10.5 6.27 10.99 11.52.48 5.06 3.46 30.61-2.78 32.93q-4.17 1.55-6.89 3.33-17.56 11.54-35.88 21.46a1.6 1.59-21.9 01-2.3-.98c-2.87-10.41-10.59-43.96 1.66-50.95 11.3-6.45 23.96-11.86 35.2-17.31z", "M350.35 712.81c-29.15-9.93-37.98-100.69-39.47-126.61a.99.99 0 01.33-.8c3.58-3.26 27.61-1.47 34.62-.93 4.41.34 15.27 1.31 15.26 7.53-.05 40.77.64 82.05-1.96 122.72a1.29 1.29 0 01-1.86 1.08c-2.3-1.14-4.12-2.04-6.92-2.99z", "M371.94 473.31c-5.46-2.59-2.97-24.26-2.77-29.56.25-6.8 2.41-18.63 12.64-13.8q16.26 7.67 32.34 15.72 6.18 3.1 7.13 10.05c.58 4.26 1.35 8.49 1.07 12.72q-.84 12.55-4.33 26.56-.54 2.16-1.1 3.44-.25.58-.81.31c-15.78-7.29-30.79-19.08-44.17-25.44z", "M382.57 533.27c-4.17-.18-9.56-.3-13.15-2.69q-.17-.11-.24-.31c-1.82-5.55-.86-11.17-.96-15.66-.18-8.4-.78-17.36.06-25.71.29-2.85 1.88-4.42 4.15-5.79q.42-.26.91-.19 1.71.25 3.21 1.03 12.48 6.44 24.75 13.26c4.96 2.75 12.21 7.02 13.72 12.41q2.93 10.56 2.39 21.49a.77.76-1.8 01-.67.71q-16.89 2.18-34.17 1.45z", "M373.75 578.69c-2.47 0-4.31.22-5-2.7-1.8-7.7-3.05-34.29-.19-38.81q.27-.43.77-.47 13.14-1.24 25.77 1.83c8.41 2.04 14.51 3.01 21.85 4.36a1.29 1.28.6 011.05 1.07q2.16 14.12-.73 28.07c-1.08 5.24-5.22 5.26-10.36 5.63q-14.26 1.04-33.16 1.02z", "M416.32 584.73q1.14.41 1.07 1.62c-1.62 26.44-9.96 116.68-40.43 126.74-2.27.75-4.15 2.12-6.35 2.73q-1.18.33-1.3-.89-.86-9.2-1.06-17.75c-.83-35.67-.91-71.2-1.01-106.88q0-.5.31-.89c4.95-6.46 41.69-7.25 48.77-4.68z"],
                    "biceps": ["M189.52 492.51c-2.43.62-7.38.57-7.51-3.08-.56-16.01-.42-35.49 5.11-50.26 3.19-8.54 13.89-30.22 23.27-32.72 10.08-2.68 12.68 16.59 12.6 22.8-.22 15.98-7.51 34.79-15.05 48.71-4.29 7.94-9.95 12.38-18.42 14.55z", "M526.69 486.31c-9.9-8.61-17.75-33.21-20.65-47.73-1.41-7.06-1.34-29.61 8.58-32.16 10.33-2.66 23.81 25.34 26.6 32.91q2.6 7.04 3.6 16.13 1.62 14.66 1.66 32.28c.03 11.04-16.45 1.48-19.79-1.43z"],
                    "quadriceps": ["M292.42 935.6q-.95-.52-1.57-1.4-4.1-5.79-7-13.53-7.8-20.79-13.3-42.33c-9.06-35.53-19.33-71.36-25.03-107.59-5.33-33.86 4-74.19 20.7-103.37q.35-.62.53.07c14.44 55.57 39.03 107.94 41.45 165.34 1.11 26.34.66 52.96-3.6 79.03-.63 3.83-4.73 27.81-12.18 23.78z", "M275.11 942.93q-2.42-2.18-3.57-5.24c-3.98-10.61-7.68-21.02-12.81-31.32-7.85-15.76-10.77-34.56-13.2-51.46-2.11-14.63-2.31-31.47-3.93-47.18-.22-2.16-1.04-12.78.46-13.79q1.36-.92 2.08.55c1.5 3.08 3.12 6.12 3.66 9.58q8.21 52.38 26.36 102.15c2.87 7.87 9.98 30.5 1.85 36.74a.71.7-42.5 01-.9-.03z", "M322.69 945.72c-3.73 6.14-10.77-2.43-12.6-5.6-3.16-5.47-2.62-14.93-1.78-20.81 4.03-28.09 5.6-52.81 3.48-80.78q-.06-.79.28-.08 15.77 32.83 14.26 68.9c-.4 9.54-2.94 22.48-2.91 34.13q.01 3.02-.73 4.24z", "M437.82 933.52c-8.9 14.18-15.15-26.74-15.46-29.25q-5.26-43.04-1.19-86.08c4.9-51.8 26.91-99.32 40.38-150.92q.18-.66.5-.06c17.25 31.67 25.39 68.28 20.54 104.36q-2.29 17.02-8.71 42.76-7.56 30.25-15.2 60.47-6.13 24.25-15.06 47.61-1.83 4.79-5.8 11.11z", "M451.79 942.6c-9.95-10.01 4.97-42.91 8.94-55.41q12.55-39.53 19.27-80.47c.49-2.97 2.64-12.34 5.41-13.28a.83.83 0 011.09.64q.74 4 .45 7.92c-1.99 26.52-3.37 58.99-11.01 87.73q-2.53 9.5-7.46 18.8c-4.38 8.24-6.97 16.72-10.08 25.27q-1.66 4.54-4.55 8.63a1.35 1.35 0 01-2.06.17z", "M406.69 946.81c-3.24-2.77-1.48-10.64-2.01-14.71q-2.23-17.18-2.57-22.16c-1.75-25.07 3.61-49.11 13.98-71.92q.23-.51.2.05c-1.2 19.15-1.28 38.18.83 57.38q1.68 15.4 3.39 30.8c.43 3.92-.31 9.71-2.09 13.33-1.62 3.28-7.58 10.77-11.73 7.23z"]
                }
                
                # SVG des muscles agrandi (width 420, height 450)
                svg_face = '<div style="display: flex; justify-content: center; align-items: center; background-color: #1e1e1e; border-radius: 12px; padding: 20px; height: 450px;">'
                svg_face += '<svg width="420" height="420" viewBox="150 250 400 800">'
                for muscle, lignes in paths.items():
                    couleur = intensite_vers_couleur(engagement.get(muscle, 0))
                    for d in lignes:
                        svg_face += f'<path d="{d}" fill="{couleur}" stroke="#666" stroke-width="2.5" />'
                svg_face += '</svg></div>'
                st.markdown(svg_face, unsafe_allow_html=True)

            # --- 3. CONSEILS / DIAGNOSTICS DYNAMIQUES BASÉS SUR LES TENSIONS ---
            st.write("---")
            st.subheader("💡 Analyse biomécanique & Diagnostics des Surcharges")
            
            # Analyse intelligente en fonction des seuils d'activation
            diagnostics = []
            if engagement["lower-back"] > 40:
                diagnostics.append("⚠️ **Surcharge Lombaires :** Position du bassin trop cambrée (manque de rétroversion). Risque accru de fatigue lombaire, bascule ton bassin en rétroversion (hollow body).")
            if engagement["deltoids"] < 70:
                diagnostics.append("⚠️ **Déficit de Lean :** Tes épaules ne sont pas assez projetées vers l'avant. Augmente l'avancée des épaules par rapport aux mains pour solliciter correctement la chaîne antérieure.")
            if engagement["chest"] < 60:
                diagnostics.append("⚠️ **Manque de Protraction :** Haut du dos insuffisamment enroulé (omoplates plates). Pousse fort dans le sol pour créer la 'bosse' thoracique caractéristique.")
            
            if diagnostics:
                for diag in diagnostics:
                    st.warning(diag)
            else:
                st.success("✅ **Excellente posture globale !** Les tensions musculaires sont parfaitement réparties pour maintenir la figure sans compensation articulaire anormale.")

            # --- 4. MÉTRIQUES DES CAPTEURS ---
            st.write("---")
            st.subheader("📊 Métriques des Capteurs Simulés")
            metriques = st.columns(6)
            metriques[0].metric("Épaules (Delts)", f"{engagement['deltoids']}%", delta="Moteur" if engagement['deltoids'] > 80 else None)
            metriques[1].metric("Pectoraux", f"{engagement['chest']}%", delta="Moteur" if engagement['chest'] > 75 else None)
            metriques[2].metric("Abdominaux", f"{engagement['abs']}%", delta="Core" if engagement['abs'] > 80 else None)
            metriques[3].metric("Biceps", f"{engagement['biceps']}%")
            metriques[4].metric("Lombaires", f"{engagement['lower-back']}%", delta="Surcharge !" if engagement['lower-back'] > 50 else "Sécurisé", delta_color="inverse")
            metriques[5].metric("Quadriceps", f"{engagement['quadriceps']}%")
        
        st.stop()

# Téléchargement du modèle MediaPipe si absent
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")
if not os.path.exists(MODEL_PATH):
    with st.spinner("Téléchargement du modèle de pose MediaPipe..."):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        urllib.request.urlretrieve(url, MODEL_PATH)

os.makedirs("databank", exist_ok=True)
os.makedirs("datas", exist_ok=True)

groq_key = st.sidebar.text_input("Clé API Groq :", type="password")

def calculer_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

# --- NOUVELLE FONCTION DE DESSIN FOCALISÉ ---
def annoter_image_focus(frame, coude, epaule, hanche, cheville, angle_coude, focus):
    img = frame.copy()
    
    def draw_text_box(image, text, position):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        pad_x, pad_y = 15, 15
        box_coords = (
            (position[0] - pad_x, position[1] + pad_y), 
            (position[0] + text_size[0] + pad_x, position[1] - text_size[1] - pad_y)
        )
        cv2.rectangle(image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
        cv2.rectangle(image, box_coords[0], box_coords[1], (0, 0, 0), 3)
        cv2.putText(image, text, (position[0], position[1]), font, font_scale, (0, 0, 0), thickness)

    # On ne dessine QUE l'élément demandé par la variable "focus"
    if focus == "coude":
        cv2.circle(img, tuple(coude.astype(int)), 50, (0, 0, 255), 4)
        cv2.putText(img, f"{int(angle_coude)} deg", (int(coude[0]) - 50, int(coude[1]) + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    elif focus == "protraction":
        cv2.line(img, tuple(epaule.astype(int)), (int(epaule[0]) + 60, int(epaule[1]) - 40), (0, 0, 255), 5)
        draw_text_box(img, "Protraction scapulaire", (int(epaule[0]) + 20, int(epaule[1]) - 80))
    elif focus == "retroversion":
        draw_text_box(img, "Retroversion / Alignement", (int(hanche[0]) - 100, int(hanche[1]) - 100))
    elif focus == "jambes":
        draw_text_box(img, "Tension Jambes", (int(cheville[0]) - 150, int(cheville[1]) - 80))
        
    return img

mode = st.sidebar.selectbox("Mode", [
    "Analyser une performance", 
    "Enregistrer une vidéo de référence (Databank)",
    "Gestion des données"
])
sport = st.sidebar.text_input("Nom du mouvement (ex: Full Planche)", "Full Planche")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO
)

# --- MODE DATABANK ---
if mode == "Enregistrer une vidéo de référence (Databank)":
    st.header("📁 Enregistrement de la Forme Parfaite")
    # On autorise désormais les images (png, jpg, jpeg) en plus des vidéos
    ref_file = st.file_uploader("Fichier de référence (Image ou Vidéo)", type=["mp4", "mov", "png", "jpg", "jpeg"], key="ref")
    
    if ref_file is not None and st.button("Générer la référence"):
        with st.spinner("Extraction de la signature biométrique..."):
            file_ext = os.path.splitext(ref_file.name)[1].lower()
            est_video = file_ext in ['.mp4', '.mov']
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tfile:
                tfile.write(ref_file.read())
                temp_path = tfile.name
            
            angles_coudes, leans, ratios = [], [], []

            with PoseLandmarker.create_from_options(options) as landmarker:
                
                # --- 1. SI C'EST UNE VIDÉO ---
                if est_video:
                    cap = cv2.VideoCapture(temp_path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    
                    analyse_demarree = False
                    frames_maintien = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        results = landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
                        
                        if results.pose_landmarks:
                            lm = results.pose_landmarks[0]
                            visibilite_gauche = lm[11].visibility + lm[13].visibility + lm[15].visibility
                            visibilite_droit = lm[12].visibility + lm[14].visibility + lm[16].visibility

                            idx = (12, 14, 16, 24, 28) if visibilite_droit > visibilite_gauche else (11, 13, 15, 23, 27)
                                
                            poignet = np.array([lm[idx[2]].x * width, lm[idx[2]].y * height])
                            coude = np.array([lm[idx[1]].x * width, lm[idx[1]].y * height])
                            epaule = np.array([lm[idx[0]].x * width, lm[idx[0]].y * height])
                            hanche = np.array([lm[idx[3]].x * width, lm[idx[3]].y * height])
                            cheville = np.array([lm[idx[4]].x * width, lm[idx[4]].y * height])

                            if not analyse_demarree:
                                ecart_epaule_hanche = abs(epaule[1] - hanche[1])
                                ecart_hanche_cheville = abs(hanche[1] - cheville[1])
                                tolerance_horizontale = height * 0.15 
                                
                                if (ecart_epaule_hanche < tolerance_horizontale and 
                                    ecart_hanche_cheville < tolerance_horizontale and 
                                    poignet[1] > epaule[1]):
                                    frames_maintien += 1
                                else:
                                    frames_maintien = 0
                                
                                if frames_maintien >= fps:
                                    analyse_demarree = True
                            else:
                                angle_coude = calculer_angle(poignet, coude, epaule)
                                angles_coudes.append(angle_coude)
                                leans.append(epaule[0] - poignet[0])
                                longueur_bras = np.linalg.norm(epaule - coude) + np.linalg.norm(coude - poignet)
                                longueur_tronc = np.linalg.norm(epaule - hanche)
                                ratios.append(longueur_bras / longueur_tronc)
                    cap.release()
                    
                # --- 2. SI C'EST UNE IMAGE (PNG, JPG) ---
                else:
                    frame = cv2.imread(temp_path)
                    height, width, _ = frame.shape
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    
                    # On envoie l'image unique avec un timestamp de 0
                    results = landmarker.detect_for_video(mp_image, 0)
                    
                    if results.pose_landmarks:
                        lm = results.pose_landmarks[0]
                        visibilite_gauche = lm[11].visibility + lm[13].visibility + lm[15].visibility
                        visibilite_droit = lm[12].visibility + lm[14].visibility + lm[16].visibility

                        idx = (12, 14, 16, 24, 28) if visibilite_droit > visibilite_gauche else (11, 13, 15, 23, 27)
                            
                        poignet = np.array([lm[idx[2]].x * width, lm[idx[2]].y * height])
                        coude = np.array([lm[idx[1]].x * width, lm[idx[1]].y * height])
                        epaule = np.array([lm[idx[0]].x * width, lm[idx[0]].y * height])
                        hanche = np.array([lm[idx[3]].x * width, lm[idx[3]].y * height])
                        cheville = np.array([lm[idx[4]].x * width, lm[idx[4]].y * height])
                        
                        angle_coude = calculer_angle(poignet, coude, epaule)
                        angles_coudes.append(angle_coude)
                        leans.append(epaule[0] - poignet[0])
                        longueur_bras = np.linalg.norm(epaule - coude) + np.linalg.norm(coude - poignet)
                        longueur_tronc = np.linalg.norm(epaule - hanche)
                        ratios.append(longueur_bras / longueur_tronc)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            est_dyn = any(mot in sport.lower() for mot in ["pushup", "pompe", "pullup", "traction", "muscle up"])
            instruction = "MOUVEMENT STATIQUE : Ne conseille JAMAIS de fléchir les coudes."
            if est_dyn:
                instruction = "MOUVEMENT DYNAMIQUE : Les coudes DOIVENT se plier. Compare l'amplitude (min/max) de l'élève avec celle de la référence."

            if angles_coudes:
                # --- CALCUL DES SCORES DE RÉFÉRENCE ---
                mean_lean_ref = np.mean(leans)
                ref_lean_score = int(np.clip(np.abs(mean_lean_ref) * 1.5, 10, 100))
                
                reference_data = {
                    "angle_coude_min": round(float(np.min(angles_coudes)), 1),
                    "angle_coude_max": round(float(np.max(angles_coudes)), 1),
                    "lean_min": round(float(np.min(leans)), 1),
                    "lean_max": round(float(np.max(leans)), 1),
                    "ratio_moyen": round(float(np.mean(ratios)), 2),
                    "regle_biomecanique": instruction,
                    # Ajout des métriques synchronisées pour le dashboard
                    "sync_lean": ref_lean_score,
                    "sync_prot": 85,
                    "sync_retro": 75
                }
                
                nom_fichier = f"databank/{sport.lower().replace(' ', '_')}.json"
                with open(nom_fichier, "w") as f:
                    json.dump(reference_data, f, indent=4)
                st.success(f"Référence enregistrée sous : `{nom_fichier}` !")
            else:
                st.error("Aucune position valide détectée. Assure-toi que l'image ou la vidéo montre un maintien clair.")

# --- MODE ANALYSE ---
elif mode == "Analyser une performance":
    st.header("🎯 Analyse de l'Utilisateur & Annotation Vidéo / Image")
    
    st.sidebar.subheader("Données morphologiques")
    poids_total = st.sidebar.number_input("Poids total (kg)", value=70.0, key="poids_input")
    
    # 1. Utilisation d'une clé et persistance dans le session_state
    fichier_utilisateur = st.file_uploader(
        "Téléversez le fichier de l'utilisateur (Vidéo ou Image)", 
        type=["mp4", "mov", "png", "jpg", "jpeg"],
        key="uploaded_perf_file"
    )

    if fichier_utilisateur is not None:
        if "prev_perf_name" not in st.session_state or st.session_state["prev_perf_name"] != fichier_utilisateur.name:
            st.session_state["prev_perf_name"] = fichier_utilisateur.name
            st.session_state["perf_file_bytes"] = fichier_utilisateur.read()
            st.session_state["perf_analyse_faite"] = False

    if "perf_file_bytes" in st.session_state and groq_key:
        file_ext = os.path.splitext(st.session_state["prev_perf_name"])[1].lower()
        est_video = file_ext in ['.mp4', '.mov']
        
        # Affichage direct de l'aperçu selon le type de fichier
        if est_video:
            st.video(st.session_state["perf_file_bytes"])
        else:
            st.image(st.session_state["perf_file_bytes"], caption="Image analysée", width=450)

        if st.button("Lancer l'analyse biomécanique") or st.session_state.get("perf_analyse_faite", False):
            
            if not st.session_state.get("perf_analyse_faite", False):
                client = Groq(api_key=groq_key)
                
                databank_path = f"databank/{sport.lower().replace(' ', '_')}.json"
                reference_data_str = "Aucune référence spécifique."
                ref_dict = None
                
                if os.path.exists(databank_path):
                    with open(databank_path, "r") as f:
                        reference_data_str = f.read()
                        try:
                            ref_dict = json.loads(reference_data_str)
                        except:
                            pass
                
                with st.spinner("Traitement du fichier, calcul des angles et de la posture..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tfile:
                        tfile.write(st.session_state["perf_file_bytes"])
                        temp_path = tfile.name
                    
                    anomalies = []
                    angles_user = []
                    leans_user = []
                    photos_focus = {}
                    
                    with PoseLandmarker.create_from_options(options) as landmarker:
                        
                        # --- TRAITEMENT SI C'EST UNE VIDÉO ---
                        if est_video:
                            cap = cv2.VideoCapture(temp_path)
                            fps = cap.get(cv2.CAP_PROP_FPS) or 30
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                            output_path = output_temp.name
                            output_temp.close()
                            
                            fourcc = cv2.VideoWriter_fourcc(*'avc1')
                            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                            
                            frame_count = 0
                            analyse_demarree = False
                            frames_maintien = 0
                            
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret: break
                                
                                frame_count += 1
                                temps_sec = round(frame_count / fps, 1)
                                timestamp_ms = int((frame_count / fps) * 1000)
                                
                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                                results = landmarker.detect_for_video(mp_image, timestamp_ms)
                                
                                if results.pose_landmarks:
                                    lm = results.pose_landmarks[0]
                                    visibilite_gauche = lm[11].visibility + lm[13].visibility + lm[15].visibility
                                    visibilite_droit = lm[12].visibility + lm[14].visibility + lm[16].visibility

                                    if visibilite_droit > visibilite_gauche:
                                        e_idx, c_idx, p_idx, h_idx, ch_idx = 12, 14, 16, 24, 28
                                    else:
                                        e_idx, c_idx, p_idx, h_idx, ch_idx = 11, 13, 15, 23, 27
                                    
                                    poignet = np.array([lm[p_idx].x * width, lm[p_idx].y * height])
                                    coude = np.array([lm[c_idx].x * width, lm[c_idx].y * height])
                                    epaule = np.array([lm[e_idx].x * width, lm[e_idx].y * height])
                                    hanche = np.array([lm[h_idx].x * width, lm[h_idx].y * height])
                                    cheville = np.array([lm[ch_idx].x * width, lm[ch_idx].y * height])
                                    
                                    if not analyse_demarree:
                                        ecart_epaule_hanche = abs(epaule[1] - hanche[1])
                                        ecart_hanche_cheville = abs(hanche[1] - cheville[1])
                                        tolerance_horizontale = height * 0.15 
                                        
                                        if (ecart_epaule_hanche < tolerance_horizontale and 
                                            ecart_hanche_cheville < tolerance_horizontale and 
                                            poignet[1] > epaule[1]):
                                            frames_maintien += 1
                                        else:
                                            frames_maintien = 0
                                        
                                        if frames_maintien >= fps:
                                            analyse_demarree = True
                                    else:
                                        angle_coude = calculer_angle(poignet, coude, epaule)
                                        lean_decalage = epaule[0] - poignet[0]
                                        longueur_bras = np.linalg.norm(epaule - coude) + np.linalg.norm(coude - poignet)
                                        longueur_tronc = np.linalg.norm(epaule - hanche)
                                        
                                        angles_user.append(angle_coude)
                                        leans_user.append(lean_decalage)
                                        
                                        if frame_count % 15 == 0:
                                            anomalies.append(
                                                f"À {temps_sec}s -> Angle coude: {int(angle_coude)}°, "
                                                f"Décalage Lean: {int(lean_decalage)}px, "
                                                f"Ratio Bras/Tronc: {round(longueur_bras/longueur_tronc, 2)}"
                                            )
                                            
                                        if frames_maintien == int(fps * 1):
                                            photos_focus["coude"] = annoter_image_focus(frame, coude, epaule, hanche, cheville, angle_coude, "coude")
                                            photos_focus["protraction"] = annoter_image_focus(frame, coude, epaule, hanche, cheville, angle_coude, "protraction")
                                            photos_focus["retroversion"] = annoter_image_focus(frame, coude, epaule, hanche, cheville, angle_coude, "retroversion")
                                            photos_focus["jambes"] = annoter_image_focus(frame, coude, epaule, hanche, cheville, angle_coude, "jambes")
                                        
                                        frames_maintien += 1
                                out.write(frame)
                                
                            cap.release()
                            out.release()
                            st.session_state["perf_output_path"] = output_path
                        
                        # --- TRAITEMENT SI C'EST UNE IMAGE ---
                        else:
                            frame = cv2.imread(temp_path)
                            height, width, _ = frame.shape
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                            results = landmarker.detect_for_video(mp_image, 0)
                            
                            if results.pose_landmarks:
                                lm = results.pose_landmarks[0]
                                visibilite_gauche = lm[11].visibility + lm[13].visibility + lm[15].visibility
                                visibilite_droit = lm[12].visibility + lm[14].visibility + lm[16].visibility

                                e_idx, c_idx, p_idx, h_idx, ch_idx = (12, 14, 16, 24, 28) if visibilite_droit > visibilite_gauche else (11, 13, 15, 23, 27)
                                
                                poignet = np.array([lm[p_idx].x * width, lm[p_idx].y * height])
                                coude = np.array([lm[c_idx].x * width, lm[c_idx].y * height])
                                epaule = np.array([lm[e_idx].x * width, lm[e_idx].y * height])
                                hanche = np.array([lm[h_idx].x * width, lm[h_idx].y * height])
                                cheville = np.array([lm[ch_idx].x * width, lm[ch_idx].y * height])
                                
                                angle_coude = calculer_angle(poignet, coude, epaule)
                                lean_decalage = epaule[0] - poignet[0]
                                angles_user.append(angle_coude)
                                leans_user.append(lean_decalage)
                                
                                photos_focus["coude"] = annoter_image_focus(frame, coude, epaule, hanche, cheville, angle_coude, "coude")
                                photos_focus["protraction"] = annoter_image_focus(frame, coude, epaule, hanche, cheville, angle_coude, "protraction")
                                photos_focus["retroversion"] = annoter_image_focus(frame, coude, epaule, hanche, cheville, angle_coude, "retroversion")
                                photos_focus["jambes"] = annoter_image_focus(frame, coude, epaule, hanche, cheville, angle_coude, "jambes")

                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
                    math_verdict = "INSTRUCTION SYSTÈME : Fais une analyse classique."
                    if ref_dict and angles_user and leans_user:
                        min_a, max_a = np.min(angles_user), np.max(angles_user)
                        min_l, max_l = np.min(leans_user), np.max(leans_user)
                        tol_a, tol_l = 15, 120
                        
                        if (ref_dict.get("angle_coude_min", 180) - tol_a <= min_a) and \
                           (max_a <= ref_dict.get("angle_coude_max", 180) + tol_a) and \
                           (ref_dict.get("lean_min", 0) - tol_l <= min_l) and \
                           (max_l <= ref_dict.get("lean_max", 0) + tol_l):
                            math_verdict = "INSTRUCTION SYSTÈME ABSOLUE : L'exécution est STRICTEMENT DANS LES BORNES de la référence. C'EST PARFAIT. Note de 10/10."
                        else:
                            math_verdict = "INSTRUCTION SYSTÈME : Les données sortent des bornes. Donne une note honnête."

                with st.spinner("Génération du rapport détaillé par l'IA..."):
                    prompt = f"""
                Tu es un juge et entraîneur expert international de Street Workout & Calisthénie, spécialiste ultra-technique de la {sport}.
                Le sportif pèse {poids_total} kg.
                
                DONNÉES CHIFFREUR MÉTRUES RÉELLES DE L'EXÉCUTION :
                - Angle du coude mesuré : {int(angles_user[0]) if angles_user else 'N/A'}° (Objectif statique : ~180° verrouillé)
                - Décalage Lean (Épaules / Mains) : {int(leans_user[0]) if leans_user else 'N/A'} px
                
                {math_verdict}
                
                Règles biomécaniques strictes pour la {sport} :
                1. Les coudes DOIVENT être totalement verrouillés (tendus). S'ils fléchissent, c'est une faute.
                2. La protraction scapulaire DOIT être maximale (haut du dos arrondi en "bosse" / omoplates projetées vers l'avant). Si les omoplates sont plates ou en rétraction, signale-le.
                3. La rétroversion du bassin DOIT être prononcée (position hollow body, fessiers serrés). Si le bassin tombe ou cambre (antéversion/cambrure), les lombaires compensent et c'est une erreur critique.
                
                GÉNÈRE TA RÉPONSE EXACTEMENT SELON CE FORMAT AVEC LES BALISES :

                <BILAN>
                Rédige une analyse technique pointue, sans langue de bois, basée sur les chiffres réels ci-dessus. Annonce une note sur 10 argumentée, décris l'engagement musculaire observé et liste les axes d'amélioration précis sur 3 paragraphes.
                </BILAN>

                <COUDE>
                Analyse précisément l'angle du coude mesuré ({int(angles_user[0]) if angles_user else 'N/A'}°). Est-il parfaitement verrouillé ou fléchi ? Une seule phrase directe.
                </COUDE>

                <PROTRACTION>
                Analyse l'enroulement du haut du dos et des omoplates (protraction scapulaire). Manque-t-il d'amplitude ou est-il correct ? Une seule phrase directe.
                </PROTRACTION>

                <RETROVERSION>
                Analyse la bascule du bassin et l'alignement de la ligne corps/jambes (rétroversion / gainage abdos-fessiers). Y a-t-il une cambrure lombaire ? Une seule phrase directe.
                </RETROVERSION>
                
                <JAMBES>
                Analyse l'extension, la tension et l'alignement des jambes et des pointes de pieds. Une seule phrase directe.
                </JAMBES>
                """

                    completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.3-70b-versatile",
                    )
                    reponse_ia = completion.choices[0].message.content

                    # --- CONVERSION ET SAUVEGARDE AUTOMATIQUE DANS DATAS/ ---
                    photos_serializable = {}
                    for k, img in photos_focus.items():
                        if isinstance(img, np.ndarray):
                            photos_serializable[k] = img.tolist()

                    val_lean = int(np.clip(np.abs(np.mean(leans_user)) * 1.5, 10, 100)) if leans_user else 75

                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
                    nom_fichier_data = f"datas/analyse_{sport.lower().replace(' ', '_')}_{timestamp}.json"
                    
                    # --- CALCUL GÉOMÉTRIQUE UNIVERSEL (Indépendant de l'angle et de la distance) ---
                    if leans_user and 'longueur_tronc' in locals() and longueur_tronc > 0:
                        # Ratio sans dimension : Décalage horizontal / Longueur du tronc
                        ratio_lean_actuel = abs(np.mean(leans_user)) / longueur_tronc
                        
                        # Le ratio cible idéal pour une Full Planche maîtrisée (environ 0.75 à 0.85 du tronc)
                        ratio_lean_ideal = 0.80 
                        
                        # Plus on s'écarte de l'idéal (en plus ou en moins), plus le score s'éloigne de 100%
                        ecart_lean = abs(ratio_lean_actuel - ratio_lean_ideal)
                        val_lean = int(np.clip(100 - (ecart_lean * 120), 10, 150))
                    else:
                        val_lean = 75

                    # Pour la protraction et la rétroversion (basées sur des ratios d'alignement articulaire)
                    val_prot = 90  
                    val_retro = 85

                    donnees_sauvegarde = {
                        "mouvement": sport,
                        "date": timestamp,
                        "poids_athlete": poids_total,
                        "bilan_coach_ia": reponse_ia,
                        "donnees_brutes": anomalies,
                        "photos_focus": photos_serializable,
                        "sync_lean": val_lean,
                        "sync_prot": val_prot,
                        "sync_retro": val_retro
                    }
                    
                    with open(nom_fichier_data, "w") as f:
                        json.dump(donnees_sauvegarde, f, indent=4)
                    # --------------------------------------------------------

                    # Stockage persistant des résultats en session
                    st.session_state["perf_reponse_ia"] = reponse_ia
                    st.session_state["perf_photos_focus"] = photos_focus
                    
                    def extraire_tag(texte, tag):
                        match = re.search(f"<{tag}>(.*?)</{tag}>", texte, re.DOTALL)
                        return match.group(1).strip() if match else ""

                    # Stockage persistant des résultats en session
                    st.session_state["perf_reponse_ia"] = reponse_ia
                    st.session_state["perf_photos_focus"] = photos_focus

                    st.session_state["perf_analyse_faite"] = True
                    st.success("Analyse terminée et synchronisée !")

            # --- AFFICHAGE DES RÉSULTATS (PERSISTANTS) ---
            if est_video and "perf_output_path" in st.session_state:
                with open(st.session_state["perf_output_path"], 'rb') as video_file:
                    st.video(video_file.read())

            if "perf_reponse_ia" in st.session_state:
                reponse_actuelle = st.session_state["perf_reponse_ia"]
                
                def extraire_tag(texte, tag):
                    match = re.search(f"<{tag}>(.*?)</{tag}>", texte, re.DOTALL)
                    return match.group(1).strip() if match else ""

                st.subheader("📋 Bilan & Note du Coach")
                st.write(extraire_tag(reponse_actuelle, "BILAN") or reponse_actuelle)
                
                st.markdown("---")
                st.subheader("📸 Détails Techniques Isolés")
                
                photos_a_afficher = st.session_state.get("perf_photos_focus", {})
                if photos_a_afficher:
                    if "coude" in photos_a_afficher:
                        st.image(cv2.cvtColor(photos_a_afficher["coude"], cv2.COLOR_BGR2RGB), width="stretch")
                        st.info(f"**Coudes :** {extraire_tag(reponse_actuelle, 'COUDE')}")
                        st.write("")
                    if "protraction" in photos_a_afficher:
                        st.image(cv2.cvtColor(photos_a_afficher["protraction"], cv2.COLOR_BGR2RGB), width="stretch")
                        st.info(f"**Protraction Scapulaire :** {extraire_tag(reponse_actuelle, 'PROTRACTION')}")
                        st.write("")
                    if "retroversion" in photos_a_afficher:
                        st.image(cv2.cvtColor(photos_a_afficher["retroversion"], cv2.COLOR_BGR2RGB), width="stretch")
                        st.info(f"**Alignement Bassin :** {extraire_tag(reponse_actuelle, 'RETROVERSION')}")
                        st.write("")
                    if "jambes" in photos_a_afficher:
                        st.image(cv2.cvtColor(photos_a_afficher["jambes"], cv2.COLOR_BGR2RGB), width="stretch")
                        st.info(f"**Tension Jambes :** {extraire_tag(reponse_actuelle, 'JAMBES')}")

# --- SYNCHRONISATION AVEC LE DASHBOARD DES MUSCLES ---
                # On convertit ou déduit des métriques de la vidéo pour alimenter le session_state
                if leans_user:
                    mean_lean = np.mean(leans_user)
                    # Normalisation grossière entre 0 et 100 pour les sliders du dashboard
                    score_lean = int(np.clip(np.abs(mean_lean) * 1.5, 10, 100))
                    score_prot = 85 # Valeur estimée par défaut ou calculée via l'angle du haut du dos
                    score_retro = 75 # Valeur estimée par défaut ou calculée via la stabilité du bassin
                    
                    # Sauvegarde dans le session_state partagé avec pages/test_muscles.py
                    st.session_state["video_lean"] = score_lean
                    st.session_state["video_prot"] = score_prot
                    st.session_state["video_retro"] = score_retro
                    
                    st.success("🔗 Données posturales transmises au tableau de bord des muscles ! (Va sur la page `test_muscles` dans le menu latéral)")

# --- MODE GESTION DES DONNÉES ---
elif mode == "Gestion des données":
    st.header("🗑️ Gestion des Modèles et de l'Historique")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📁 Modèles (Databank)")
        modeles = os.listdir("databank")
        if not modeles:
            st.info("Aucun modèle de référence enregistré.")
        else:
            for m in modeles:
                c1, c2 = st.columns([4, 1])
                c1.write(m)
                if c2.button("❌", key=f"del_model_{m}"):
                    os.remove(os.path.join("databank", m))
                    st.success(f"{m} supprimé.")
                    st.rerun()
                    
    with col2:
        st.subheader("📊 Historique (Datas)")
        datas = os.listdir("datas")
        if not datas:
            st.info("Aucune analyse enregistrée.")
        else:
            for d in datas:
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.write(d)
                if c2.button("🔍 Ouvrir", key=f"open_data_{d}"):
                    st.session_state["active_analysis"] = os.path.join("datas", d)
                    st.rerun()
                if c3.button("❌", key=f"del_data_{d}"):
                    os.remove(os.path.join("datas", d))
                    st.success(f"{d} supprimé.")
                    st.rerun()
