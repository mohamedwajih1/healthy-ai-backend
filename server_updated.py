from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import threading
import time
from datetime import datetime
from behavioral_ai_fixed_clean import BehavioralAIEngine
from train_model import HabitModelTrainer
from rl_engine_numpy import RLEngine, get_rl_engine, SUGGESTIONS, get_suggestion_text, detect_missing_categories
import traceback

# Windows consoles can default to cp1252 which crashes on Arabic prints.
# Keep behavior unchanged; only stabilize debug/trace output.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return {
        "message": "AI Backend is running with RL support",
        "endpoints": [
            "/health",
            "/analyze",
            "/smart_suggest",
            "/smart_suggest_rl",
            "/feedback",
            "/rl_stats",
            "/rl_train"
        ],
        "features": [
            "rule_based_suggestions",
            "rl_q_learning",
            "feedback_loop",
            "experience_replay"
        ]
    }

# Enable CORS for all origins (development mode)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize components
behavioral_engine = None
model_trainer = None
rl_engine = None

# Auto-learning tracking
learning_data = []
retrain_threshold = 1000
last_retrain_count = 0
last_retrain_date = None

# RL Feedback tracking
rl_feedback_buffer = []
rl_last_training = time.time()
RL_TRAINING_INTERVAL = 300  # 5 minutes between training sessions

def initialize_rl_engine():
    """Initialize RL Engine"""
    global rl_engine
    try:
        rl_engine = get_rl_engine()
        print(f"✅ RL Engine initialized: epsilon={rl_engine.epsilon:.3f}")
        return True
    except Exception as e:
        print(f"❌ RL Engine initialization failed: {e}")
        return False

# 🗺 1️⃣ One-Hot Suggestion Mapping
SUGGESTIONS = [
    'drink_water', 'exercise', 'sleep_early', 'meditate', 
    'read_book', 'healthy_food', 'walk', 'stretch', 
    'deep_work', 'no_sugar'
]

def get_suggestion_vector(name):
    """One-Hot Encoding to prevent fake ordering (Bomb #1 Fixed)"""
    vector = [0] * len(SUGGESTIONS)
    for i, s in enumerate(SUGGESTIONS):
        if s in name.lower():
            vector[i] = 1
            break
    return vector

def initialize_system():
    """Initialize AI habit analysis system with behavioral AI engine"""
    global behavioral_engine, model_trainer
    
    try:
        # Train model if not exists
        if not os.path.exists('habit_model.pkl'):
            print("Training new model...")
            model_trainer = HabitModelTrainer()
            model_trainer.train()
        
        # Initialize behavioral AI engine
        behavioral_engine = BehavioralAIEngine()
        
        # Initialize RL Engine
        initialize_rl_engine()
        
        print("✅ All AI systems initialized successfully")
        return True
    except Exception as e:
        print(f"System initialization failed: {str(e)}")
        return False

# AUTO INIT FOR PRODUCTION (Render / Gunicorn)
# When running with gunicorn, __main__ block won't execute
# So we auto-initialize when module loads if not already done
if behavioral_engine is None:
    print("🔄 Auto-initializing system (production mode)...")
    initialize_system()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'system_initialized': behavioral_engine is not None
    })

@app.route('/analyze', methods=['POST'])
def analyze_habits():
    """Main habit analysis endpoint with intelligent Arabic AI"""
    try:
        print("=== FLASK DEBUG: /analyze REQUEST RECEIVED ===")
        if not behavioral_engine:
            return jsonify({
                'error': 'System not initialized',
                'message': 'Please wait for system initialization'
            }), 500
        
        # Get data from request
        data = request.get_json()
        print("=== RECEIVED DATA ===", data)
        print("=== RAW INPUT FROM FLUTTER ===")
        print(f"RAW INPUT: {data}")
        print("============================")
        
        # Validate required fields - completion_rate is the only one that matters
        required_fields = ['completion_rate', 'activeStreaks', 'bestStreak', 'totalHabits', 'consistency', 'dropRate']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'details': [f'{field} is required']
                }), 400
        
        # Validate field types and apply safety normalization
        try:
            # Extract completion rate and other metrics from request
            # IGNORE completionRate completely - USE completion_rate ONLY
            completion = float(data.get('completion_rate', 0.0))
            print("USED completion_rate:", completion)
            
            consistency = float(data.get('consistency', 0.0))
            dropRate = float(data.get('dropRate', 0.0))
            activeStreaks = float(data.get('activeStreaks', 0.0))
            bestStreak = float(data.get('bestStreak', 0.0))
            totalHabits = float(data.get('totalHabits', 0.0))
        except (ValueError, TypeError) as e:
            return jsonify({
                'error': 'All numeric fields must be valid numbers',
                'details': [str(e)]
            }), 400
        
        # Apply safety normalization - clamp all values to valid ranges
        completion = max(0.0, min(1.0, completion))
        consistency = max(0.0, min(1.0, consistency))
        dropRate = max(0.0, min(1.0, dropRate))
        activeStreaks = max(0.0, activeStreaks)  # Can't be negative
        bestStreak = max(0.0, bestStreak)  # Can't be negative
        totalHabits = max(0.0, totalHabits)  # Can't be negative
        
        # STEP 1: FORCE SYNC - IMMEDIATELY UPDATE REQUEST DATA
        data['completion_rate'] = completion
        
        # STEP 4: MANDATORY DEBUG - SHOW REQUEST VS USED COMPLETION
        print("=== FINAL COMPLETION SYNC ===")
        print("Request completion_rate:", data.get("completion_rate"))
        print("Used completion:", completion)
        print("================================")
        
        # Update data with normalized values
        normalized_data = {
            'completionRate': completion,  # Keep key for AI engine but use completion value
            'activeStreaks': activeStreaks,
            'bestStreak': bestStreak,
            'totalHabits': totalHabits,
            'consistency': consistency,
            'dropRate': dropRate
        }
        
        print("=== NORMALIZED INPUT ===")
        print(f"NORMALIZED DATA: {normalized_data}")
        print("========================")
        
        # STEP 5: SAFETY DEBUG - SHOW RAW VS NORMALIZED
        print("=== FINAL COMPLETION VALUE ===")
        print("RAW completion_rate:", data.get('completion_rate', 'NOT_SET'))
        print("NORMALIZED completion:", completion)
        print("================================")
        
        # STEP 3: SERVER VALIDATION
        print("==== SERVER RECEIVED ====");
        print(data)
        
        # Extract habit names if provided
        habit_names = data.get('habit_names', [])
        completed_habits = data.get('completed_habits', [])
        completion_rate = completion
        
        print("habit_names:", habit_names)
        print("completed_habits:", completed_habits)
        print("completion_rate:", completion_rate)
        
        # ENSURE: completion_rate != 0, completed_habits != []
        if completion_rate == 0.0:
            print("WARNING: completion_rate is 0 - BUG IN FLUTTER")
        if not completed_habits:
            print("WARNING: completed_habits is empty - BUG IN FLUTTER")
        print("consistency:", data.get('consistency', 0))
        
        if not isinstance(habit_names, list):
            return jsonify({
                'error': 'habitNames must be an array'
            }), 400
        
        # Generate insights using behavioral AI engine
        print("=== SENDING TO BEHAVIORAL AI ENGINE ===")
        print(f"FEATURES FOR DETECTION: {normalized_data}")
        print("=====================================")
        
        print("FINAL INPUT TO AI:")
        print(normalized_data, habit_names, completed_habits)
        
        # STEP 1: INTELLIGENT RULE LAYER ABOVE MODEL
        print("=== INTELLIGENT RULE LAYER ===")
        
        # STEP 1: DETECT MISSING
        missing_habits = [h for h in habit_names if h not in completed_habits]
        print(f"Missing habits: {missing_habits}")
        
        # Generate AI insights (for summary/behaviors only, NOT suggestions)
        insights = behavioral_engine.generate_insights(normalized_data, habit_names, completed_habits)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: ADAPTIVE HABIT SUGGESTION ENGINE 🤖
        # ═══════════════════════════════════════════════════════════════
        
        from datetime import datetime
        import random
        
        # 🔍 1. GAP DETECTION - اكتشاف الفجوات السلوكية
        habit_categories = {
            "physical": {
                "habits": ["water", "exercise", "sleep", "walk"],
                "suggestions": [
                    "💧 اشرب كوب ماء الآن (ابدأ ببساطة)",
                    "🚶‍♂️ امشِ 10 دقائق في الهواء الطلق",
                    "😴 حدد موعد نوم ثابت الليلة",
                    "🤸 جرب تمارين التمدد 5 دقائق"
                ]
            },
            "mental": {
                "habits": ["thinking", "reading", "journaling", "breathing"],
                "suggestions": [
                    "🧘 جرب 3 دقائق تنفس عميق",
                    "📖 اقرأ صفحة واحدة قبل النوم",
                    "✍️ اكتب 3 أشياء ممتن لها اليوم",
                    "🎯 خطط لأولوية واحدة غداً"
                ]
            },
            "productivity": {
                "habits": ["focus_time", "screen_limit", "morning_routine", "planning"],
                "suggestions": [
                    "⏱️ استخدم تقنية Pomodoro (25 دقيقة تركيز)",
                    "📱 أبعد الهاتف عنك لمدة ساعة",
                    "🌅 ابدأ يومك بـ 5 دقائق تخطيط",
                    "📝 رتب مهامك حسب الأهمية"
                ]
            }
        }
        
        # تحديد الفئات الموجودة عند المستخدم
        user_categories = set()
        for habit in habit_names:
            for cat, data in habit_categories.items():
                if habit in data["habits"]:
                    user_categories.add(cat)
        
        # اقتراحات من الفئات الناقصة
        missing_categories = set(habit_categories.keys()) - user_categories
        gap_suggestions = []
        for cat in missing_categories:
            gap_suggestions.extend(habit_categories[cat]["suggestions"])
        random.shuffle(gap_suggestions)
        
        # 🔄 2. COMPENSATORY SUGGESTIONS - اقتراحات مكافئة للتراجع
        compensatory_suggestions = []
        if dropRate > 0.35:
            compensatory_suggestions = [
                "🆘 ابدأ بـ عادة واحدة فقط هذا الأسبوع",
                "💪 اختر أسهل عادة عندك وأكملها 3 أيام",
                "🎉 احتفل بأي إنجاز صغير (حتى لو 50%)",
                "🤝 ابحث عن صديق يشاركك العادة"
            ]
        elif dropRate > 0.2:
            compensatory_suggestions = [
                "📊 راجع: متى بدأ التراجع؟ (حدد المشكلة)",
                "⏰ جرب وقت مختلف للعادة (صباح/مساء)",
                "🔔 ضبط تذكير جديد أكثر وضوحاً"
            ]
        
        # 🕐 3. TIME-BASED SUGGESTIONS - اقتراحات حسب الوقت
        hour = datetime.now().hour
        time_suggestions = []
        if 5 <= hour < 12:  # الصباح
            time_suggestions = [
                "🌅 اشرب ماء فور استيقاظك",
                "🧘 5 دقائق تأمل قبل الهاتف",
                "📝 اكتب هدف واحد لليوم",
                "🏃‍♂️ تمارين خفيفة لتنشيط الجسم"
            ]
        elif 12 <= hour < 17:  # الظهر
            time_suggestions = [
                "🚶‍♂️ استراحة مشي 5 دقائق",
                "💧 اشرب كوب ماء الآن",
                "👀 راحة عين 20 ثانية (ابعد عن الشاشة)",
                "🧘 تنفس عميق 3 مرات"
            ]
        elif 17 <= hour < 21:  # المساء المبكر
            time_suggestions = [
                "📖 اقرأ 10 دقائق قبل النوم",
                "🌙 جهز ملابسك لليوم التالي",
                "📵 أبعد الهاتف 30 دقيقة قبل النوم",
                "✍️ راجع يومك بـ 3 أسطر"
            ]
        else:  # الليل
            time_suggestions = [
                "😴 نم الآن! (النوم أولوية قصوى)",
                "🌙 استرخِ بـ 4-7-8 تنفس",
                "📝 اكتب مخاوفك (أفرغ عقلك)",
                "🛁 دش دافئ سريع"
            ]
        
        # 🔥 4. BROKEN STREAK RECOVERY - استرداد السلاسل المكسورة
        streak_recovery = []
        # نحتاج best_streak من features - نستخدم normalized_data
        best_streak_val = int(normalized_data.get('bestStreak', 0))
        active_streaks_val = normalized_data.get('activeStreaks', 0)
        
        if best_streak_val >= 3 and active_streaks_val < 0.3:
            streak_recovery = [
                f"🔥 كنت واصل {best_streak_val} أيام! عاود من جديد",
                "🎯 اليوم = يوم 1 جديد (لا تندم)",
                "💪 راجع السلسلة السابقة: ماذا نجح؟",
                "🏆 حدد مكافأة لـ 3 أيام متتالية"
            ]
        
        # 🎯 5. DIVERSITY ENGINE + SMART SELECTION
        all_candidates = gap_suggestions + compensatory_suggestions + time_suggestions + streak_recovery
        
        # إضافة اقتراحات من العادات الناقصة التقليدية
        traditional_missing = []
        traditional_map = {
            "water": "💧 اشرب كمية كافية من الماء",
            "reading": "📚 اقرأ يوميًا حتى صفحة واحدة",
            "exercise": "🏃 مارس رياضة خفيفة 10 دقائق",
            "sleep": "😴 نظم نومك (7-8 ساعات)",
            "thinking": "� خذ وقت للتفكر 5 دقائق"
        }
        for h in missing_habits:
            if h in traditional_map and traditional_map[h] not in all_candidates:
                traditional_missing.append(traditional_map[h])
        
        all_candidates.extend(traditional_missing)
        
        # اختيار متنوع (من فئات مختلفة)
        selected_suggestions = []
        used_prefixes = set()  # تتبع الفئات المستخدمة
        
        for suggestion in all_candidates:
            # استخراج الفئة من الأيقونة (أول رمز إيموجي)
            prefix = suggestion[:2] if suggestion[:2] in ["💧", "📚", "🏃", "😴", "🧘", "🎯", "🆘", "🕐", "🔥", "⏰", "📝", "🌅", "🚶", "👀", "🌙", "📵", "🤝", "💪", "📊"] else "other"
            
            if prefix not in used_prefixes or len(selected_suggestions) < 2:
                selected_suggestions.append(suggestion)
                used_prefixes.add(prefix)
            
            if len(selected_suggestions) >= 4:
                break
        
        # إذا لسه محتاجين اقتراحات، نضيف من الـ AI أو العامة
        while len(selected_suggestions) < 4:
            ai_fallback = [
                "🎯 ركز على عادة واحدة هذا الأسبوع",
                "📊 راقب تقدمك يومياً (حتى لو صغير)",
                "🎉 احتفل بأي إنجاز (تعزيز إيجابي)",
                "🤔 سأل نفسك: لماذا هذه العادة مهمة؟"
            ]
            for fb in ai_fallback:
                if fb not in selected_suggestions:
                    selected_suggestions.append(fb)
                    break
        
        final_suggestions = selected_suggestions[:4]
        
        # DEBUG: Adaptive Engine Stats
        print("=== ADAPTIVE HABIT ENGINE ===")
        print(f"🎯 User categories: {user_categories}")
        print(f"🔍 Missing categories: {missing_categories}")
        print(f"⏰ Current hour: {hour} → Time-based active")
        print(f"📉 Drop rate: {dropRate:.2f} → Compensatory: {len(compensatory_suggestions)} suggestions")
        print(f"🔥 Streak recovery: {len(streak_recovery)} suggestions")
        print(f"📊 Total candidates: {len(all_candidates)}")
        print(f"✅ Final (diverse): {final_suggestions}")
        
        # STEP 3: SMART MERGE (AI + Rule balanced)
        ai_suggestions = insights.get('suggestions', [])
        
        # Shuffle AI suggestions to reduce repetition across requests
        import random
        random.shuffle(ai_suggestions)
        
        def smart_merge(ai_sugs, rule_sugs, completion_rate, max_count=4):
            """
            Merge AI + Rule suggestions intelligently based on user state
            """
            merged = []
            used = set()

            # Normalize inputs
            ai_clean = [s.strip() for s in ai_sugs if s and len(s.strip()) > 3]
            rule_clean = [s.strip() for s in rule_sugs if s and len(s.strip()) > 3]

            # AI-LED PRIORITY: AI leads, rules supplement
            # Only use rule-first for very low completion (needs basics)
            if completion_rate < 0.3:
                # Very low performance: basics first, then AI
                primary = rule_clean
                secondary = ai_clean
            else:
                # AI leads for all other cases (30%+ completion)
                primary = ai_clean
                secondary = rule_clean

            # 🧠 دمج ذكي
            i = j = 0
            while len(merged) < max_count:
                if i < len(primary):
                    s = primary[i]
                    if s not in used:
                        merged.append(s)
                        used.add(s)
                    i += 1

                if len(merged) >= max_count:
                    break

                if j < len(secondary):
                    s = secondary[j]
                    if s not in used:
                        merged.append(s)
                        used.add(s)
                    j += 1

                if i >= len(primary) and j >= len(secondary):
                    break

            return merged[:max_count]
        
        merged = smart_merge(ai_suggestions, final_suggestions, completion_rate, max_count=4)
        insights['suggestions'] = merged
        
        # DEBUG
        print("=== AI-LED MERGE ===")
        print(f"AI first with {len(ai_suggestions)} suggestions, rules supplement")
        print("AI suggestions:", ai_suggestions[:4])
        print("Rule suggestions:", final_suggestions[:4])
        print("Merged:", merged)
        
        # STEP 4: DYNAMIC TONE WITH SUB-CONDITIONS
        ai_summary = insights.get('summary', '')
        
        if completion_rate == 0.0:
            dynamic_summary = "واضح إنك لم تبدأ الالتزام بعد، حاول تركز على خطوة واحدة في كل مرة"
        elif completion_rate < 0.4:
            if dropRate > 0.3:
                dynamic_summary = f"أداؤك ضعيف ({completion_rate*100:.0f}%) وعندك تراجع واضح، لازم توقف النزيف ده وتبدأ تلتزم"
            elif consistency < 0.3:
                dynamic_summary = f"أداؤك ضعيف ({completion_rate*100:.0f}%) وغير مستقر، ركز على عادة واحدة بس كل يوم"
            else:
                dynamic_summary = f"أداؤك ضعيف ({completion_rate*100:.0f}%)، تحتاج تركز أكتر وتلتزم بالعادات الأساسية"
        elif 0.4 <= completion_rate < 0.8:
            if dropRate > 0.3:
                focus_text = "بس عندك تراجع في الأيام الأخيرة" if missing_habits else "بس عندك تراجع في الأيام الأخيرة"
                dynamic_summary = f"أنت ماشي كويس ({completion_rate*100:.0f}%)، {focus_text}، حاول تستقر أكتر"
            elif consistency < 0.4:
                focus_text = "حاول تركز على العادات اللي نسيتها اليوم" if missing_habits else "أداؤك متغير يومياً، محتاج روتين أثبت"
                dynamic_summary = f"أنت ماشي كويس ({completion_rate*100:.0f}%)، {focus_text}"
            elif activeStreaks > 0.6:
                dynamic_summary = f"أنت ماشي كويس ({completion_rate*100:.0f}%) وعندك سلاسل التزام حلوة، كمل كده"
            else:
                focus_text = "حاول تركز على العادات اللي نسيتها اليوم" if missing_habits else "أداؤك مقبول ولكن يحتاج تحسين"
                dynamic_summary = f"أنت ماشي كويس ({completion_rate*100:.0f}%)، {focus_text}"
        else:  # completion_rate >= 0.8
            if consistency > 0.7:
                dynamic_summary = f"أداؤك ممتاز ({completion_rate*100:.0f}%) وثابت 👏، حافظ على المستوى ده وفكر في تحديات جديدة"
            elif dropRate > 0.2:
                dynamic_summary = f"أداؤك ممتاز ({completion_rate*100:.0f}%) بس انتبه التراجع، حافظ على الزخم"
            else:
                dynamic_summary = f"أداؤك ممتاز ({completion_rate*100:.0f}%)، حافظ على هذا المستوى وفكر في تحديات جديدة"
        
        # AI-First Summary Logic
        # Use AI summary as base, enhance it with context, fallback to dynamic only if needed
        ai_summary = insights.get('summary', '').strip()
        
        if ai_summary:
            # AI provided a summary - use it as base with light enhancement
            summary = ai_summary
            
            # Add contextual hint based on performance (not replacement, just enhancement)
            if completion_rate < 0.4 and "حاول" not in summary and "ابدأ" not in summary:
                summary += " — حاول تبدأ بخطوة صغيرة اليوم"
            elif completion_rate > 0.8 and "ممتاز" not in summary and "استمر" not in summary:
                summary += " — ممتاز، استمر وطور نفسك أكثر"
        else:
            # AI returned empty - use dynamic summary as fallback
            summary = dynamic_summary
            print("WARNING: AI summary empty, using dynamic fallback")
        
        insights['summary'] = summary
        
        # DEBUG OUTPUT
        print("==== AI RESPONSE ====");
        print(f"AI Original: {ai_summary[:50] if ai_summary else 'EMPTY'}...")
        print(f"Final Summary: {summary[:60]}...")
        
        # Log detected behaviors
        detected_behaviors = insights.get('detectedBehaviors', [])
        print("=== BEHAVIOR DETECTION RESULTS ===")
        print(f"DETECTED BEHAVIORS: {detected_behaviors}")
        
        if not detected_behaviors:
            print("WARNING: No behaviors detected!")
        else:
            print(f"SUCCESS: Detected {len(detected_behaviors)} behavior(s)")
        
        print("=====================================")
        
        # Safety check - should never happen now but keep as extra safety
        if not summary or summary.strip() == "":
            summary = dynamic_summary
            print("EMERGENCY FALLBACK: Using dynamic summary")
        
        print("FINAL SUMMARY:", summary)
        
        safe_insights = {
            'summary': summary,
            'strengths': insights.get('strengths') or [],
            'improvements': insights.get('improvements') or [],
            'suggestions': merged[:4],
            'recommendedHabits': insights.get('recommendedHabits') or [],
            'userScore': insights.get('userScore') or 0.0,
            'userState': insights.get('userState') or "غير معروف",
            'trend': insights.get('trend') or "مستقر",
            'confidence': insights.get('confidence') or 0.0,
            'futureScore': insights.get('futureScore') or 0.0,
            'detectedBehaviors': insights.get('detectedBehaviors') or []
        }
        
        # MERGE OVERRIDE - safe_insights
        safe_insights['suggestions'] = merged[:4]
        
        # HARD OVERRIDE - final response
        result = {
            'success': True,
            'data': {
                'summary': safe_insights['summary'],
                'strengths': safe_insights['strengths'],
                'improvements': safe_insights['improvements'],
                'suggestions': merged[:4],
                'recommendedHabits': safe_insights['recommendedHabits'],
                'userScore': safe_insights['userScore'],
                'userState': safe_insights['userState'],
                'trend': safe_insights['trend'],
                'confidence': safe_insights['confidence'],
                'futureScore': safe_insights['futureScore'],
                'detectedBehaviors': safe_insights['detectedBehaviors']
            },
            'timestamp': str(os.times())
        }
        
        # STEP 4: FINAL GUARD (ANTI-LEAK)
        assert safe_insights['suggestions'] == merged[:4]
        assert result['data']['suggestions'] == merged[:4]
        
        return jsonify(result)
        
    except Exception as e:
        error_details = traceback.format_exc()
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e),
            'details': error_details if app.debug else None
        }), 500

@app.route('/track_interaction', methods=['POST'])
def track_interaction():
    """Track user interaction with AI suggestions - FOR LEARNING"""
    global learning_data
    
    try:
        data = request.get_json()
        
        interaction = {
            'userId': data.get('userId'),
            'timestamp': datetime.now().isoformat(),
            'features': data.get('features'),  # User state when suggestion given
            'suggestion': data.get('suggestion'),  # What AI suggested
            'executed': data.get('executed', False),  # Did user do it?
            'feedback': data.get('feedback', 0),  # 1 = good, -1 = bad
            'context': data.get('context')  # Time, day, etc.
        }
        
        learning_data.append(interaction)
        
        # Check if we should retrain
        should_retrain = False
        if len(learning_data) - last_retrain_count >= retrain_threshold:
            should_retrain = True
        
        # Also check weekly retrain
        if last_retrain_date:
            days_since = (datetime.now() - last_retrain_date).days
            if days_since >= 7:
                should_retrain = True
        
        return jsonify({
            'success': True,
            'tracked': len(learning_data),
            'should_retrain': should_retrain,
            'message': 'Interaction tracked for AI learning'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    """Retrain model on accumulated real data"""
    global behavioral_engine, last_retrain_count, last_retrain_date
    
    try:
        if len(learning_data) < 100:
            return jsonify({
                'success': False,
                'message': 'Need at least 100 interactions to retrain',
                'current': len(learning_data)
            }), 400
        
        print(f"🔄 Retraining AI on {len(learning_data)} real interactions...")
        
        # 2️⃣ Prepare Features (Bomb #2 & #3 Fixed)
        for interaction in learning_data:
            if interaction.get('features') and interaction.get('suggestion'):
                feat = interaction['features']
                
                # Context Awareness (Time/Day)
                dt = datetime.fromisoformat(interaction.get('timestamp', datetime.now().isoformat()))
                hour_norm = dt.hour / 24.0
                day_norm = dt.weekday() / 7.0
                
                # Behavioral Personalization (Better than Hash)
                user_avg_comp = feat.get('completionRate', 0.5) 
                user_consistency = feat.get('consistency', 0.5)
                
                # One-Hot Encoding for Suggestion
                s_vector = get_suggestion_vector(interaction['suggestion'])
                
                # Full Feature Vector
                features_vector = [
                    feat.get('completionRate', 0),
                    feat.get('consistency', 0),
                    feat.get('dropRate', 0),
                    feat.get('activeStreaks', 0),
                    hour_norm,     # 🎯 Time Awareness
                    day_norm,      # 🎯 Day Awareness
                    user_avg_comp, # 🎯 Personalization
                    user_consistency
                ] + s_vector # 🎯 One-Hot (No Bias)
                
                # 🎯 Preference Learning (Boost chosen, slight penalty for ignored)
                chosen_suggestion = interaction.get('chosenSuggestion')
                current_s = interaction.get('suggestion')
                
                # Base Reward calculation:
                feedback = interaction.get('feedback', 0)
                executed = interaction.get('executed', False)
                
                if executed and feedback > 0: reward = 1.0
                elif executed: reward = 0.7
                elif not executed and feedback < 0: reward = -0.5
                else: reward = 0.2
                
                # 🔥 Preference Learning Adjustment
                if chosen_suggestion:
                    if chosen_suggestion == current_s:
                        reward += 0.2  # 🚀 User preferred this!
                    else:
                        reward -= 0.1  # 📉 User ignored this for another choice
                
                reward = max(-1.0, min(1.2, reward)) # Clamp rewards
                
                X.append(features_vector)
                y.append(reward)
        
        if len(X) < 50:
            return jsonify({'success': False, 'message': 'Not enough data'}), 400
        
        # 3️⃣ Regularized Regressor (Bomb #3: Prevents Overfitting)
        from sklearn.ensemble import RandomForestRegressor
        reward_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,           # 🔥 Limit depth
            min_samples_leaf=12,   # 🔥 Require more samples
            random_state=42
        )
        reward_model.fit(X_scaled, y)
        
        print(f"✅ Reward model trained: predicts success score 0.0-1.0")
        print(f"   Average reward in training: {sum(y)/len(y):.2f}")
        
        # Save models
        joblib.dump(reward_model, 'reward_model_real.pkl')
        joblib.dump(scaler, 'feature_scaler_real.pkl')
        
        # Also train suggestion picker model
        train_suggestion_picker(X_scaled, y, learning_data)
        
        # Update behavioral engine
        if behavioral_engine:
            behavioral_engine.reward_model = reward_model
            behavioral_engine.scaler = scaler
        
        # Update tracking
        last_retrain_count = len(learning_data)
        last_retrain_date = datetime.now()
        
        # Save learning data to file
        with open('learning_data.json', 'w', encoding='utf-8') as f:
            json.dump(learning_data, f, ensure_ascii=False)
        
        print(f"✅ Retraining complete! Model now uses {len(X)} real examples")
        
        return jsonify({
            'success': True,
            'trained_on': len(X),
            'total_interactions': len(learning_data),
            'message': 'AI model retrained on real user data'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/learning_status', methods=['GET'])
def learning_status():
    """Get AI learning status"""
    try:
        executed_count = sum(1 for i in learning_data if i.get('executed'))
        positive_feedback = sum(1 for i in learning_data if i.get('feedback', 0) > 0)
        negative_feedback = sum(1 for i in learning_data if i.get('feedback', 0) < 0)
        
        # Calculate average reward from learning data
        avg_reward = 0
        if learning_data:
            rewards = []
            for i in learning_data:
                executed = i.get('executed', False)
                feedback = i.get('feedback', 0)
                if executed and feedback > 0:
                    rewards.append(1.0)
                elif executed:
                    rewards.append(0.7)
                elif feedback < 0:
                    rewards.append(0.0)
                else:
                    rewards.append(0.3)
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
        
        return jsonify({
            'total_interactions': len(learning_data),
            'executed_suggestions': executed_count,
            'positive_feedback': positive_feedback,
            'negative_feedback': negative_feedback,
            'average_reward_score': round(avg_reward, 3),
            'retrain_threshold': retrain_threshold,
            'until_next_retrain': retrain_threshold - (len(learning_data) - last_retrain_count),
            'last_retrain': last_retrain_date.isoformat() if last_retrain_date else None,
            'model_type': 'reward_regressor' if os.path.exists('reward_model_real.pkl') else 'synthetic',
            'learning_mode': 'reinforcement_learning',
            'suggestion_picker': 'AI-optimized'
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

# SMART SUGGESTION PICKER - learns which suggestion works for whom
def train_suggestion_picker(X_scaled, rewards, learning_data):
    """Train model to pick the best suggestion based on user state"""
    from collections import defaultdict
    
    # Group by suggestion and calculate average reward
    suggestion_stats = defaultdict(list)
    for i, interaction in enumerate(learning_data):
        if i < len(rewards):
            suggestion = interaction.get('suggestion', 'unknown')
            suggestion_stats[suggestion].append(rewards[i])
    
    # Calculate average effectiveness
    suggestion_effectiveness = {}
    for suggestion, rewards_list in suggestion_stats.items():
        avg_reward = sum(rewards_list) / len(rewards_list)
        suggestion_effectiveness[suggestion] = {
            'avg_reward': avg_reward,
            'count': len(rewards_list)
        }
    
    # Save effectiveness data
    joblib.dump(suggestion_effectiveness, 'suggestion_effectiveness.pkl')
    
    print(f"📊 Suggestion effectiveness trained on {len(suggestion_stats)} suggestions")
    for suggestion, stats in sorted(suggestion_effectiveness.items(), 
                                     key=lambda x: x[1]['avg_reward'], reverse=True)[:5]:
        print(f"   {suggestion}: {stats['avg_reward']:.2f} avg reward ({stats['count']} uses)")
    
    return suggestion_effectiveness

@app.route('/smart_suggest', methods=['POST'])
def smart_suggest():
    """Predicts Top 3 Suggestions using Context & Personalization"""
    import random
    import numpy as np
    
    try:
        data = request.get_json()
        features = data.get('features', {})
        available_suggestions = data.get('available_suggestions', [])
        
        # Current Context
        now = datetime.now()
        hour_norm = now.hour / 24.0
        day_norm = now.weekday() / 7.0
        
        if not available_suggestions:
            return jsonify({'error': 'No suggestions provided'}), 400
            
        # 1️⃣ Load Reward Model
        model, scaler = None, None
        if os.path.exists('reward_model_real.pkl'):
            model = joblib.load('reward_model_real.pkl')
            scaler = joblib.load('feature_scaler_real.pkl')
        
        # 2️⃣ Predict Score for EVERY available suggestion
        scored_list = []
        for suggestion in available_suggestions:
            if model and scaler:
                s_vector = get_suggestion_vector(suggestion)
                x = [
                    features.get('completionRate', 0.5),
                    features.get('consistency', 0.5),
                    features.get('dropRate', 0),
                    features.get('activeStreaks', 0),
                    hour_norm,
                    day_norm,
                    features.get('completionRate', 0.5), # user avg
                    features.get('consistency', 0.5)     # user consistency
                ] + s_vector
                
                x_scaled = scaler.transform([x])
                score = float(model.predict(x_scaled)[0])
            else:
                score = 0.5 # Cold start fallback
            
            scored_list.append({'suggestion': suggestion, 'score': score})
            
        # 3️⃣ Ranking & Dynamic Sampling (Top 3 from Top 5) 🔥
        scored_list.sort(key=lambda x: x['score'], reverse=True)
        top_5 = scored_list[:5]
        
        # 🎲 Dynamic Selection: Pick 3 from the top 5 to keep it fresh
        if len(top_5) >= 3:
            top_3 = random.sample(top_5, 3)
            # Re-sort the sampled 3 by score
            top_3.sort(key=lambda x: x['score'], reverse=True)
        else:
            top_3 = top_5
        
        # Determine the winner
        winner = top_3[0]
        method = "Preference-Aware Ranking"
            
        return jsonify({
            'main_suggestion': winner['suggestion'],
            'predicted_reward': round(winner['score'], 3),
            'top_suggestions': [
                {'suggestion': s['suggestion'], 'score': round(s['score'], 3)} 
                for s in top_3
            ],
            'method': method,
            'confidence': "Personalized" if model is not None else "Learning",
            'note': "✨ أفضل اقتراح لك الآن بناءً على تفضيلاتك"
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# Auto-retrain scheduler (runs every hour)
def check_and_retrain():
    """Background task to check if retraining needed"""
    global last_retrain_count, last_retrain_date
    
    try:
        current_count = len(learning_data)
        new_records = current_count - last_retrain_count
        
        # Check threshold
        if new_records >= retrain_threshold:
            print(f"🔄 Auto-triggering retrain: {new_records} new records")
            with app.test_client() as client:
                client.post('/retrain_model')
            return
        
        # Check weekly
        if last_retrain_date:
            days_since = (datetime.now() - last_retrain_date).days
            if days_since >= 7 and new_records >= 100:
                print(f"🔄 Weekly retrain triggered")
                with app.test_client() as client:
                    client.post('/retrain_model')
    except Exception as e:
        print(f"Auto-retrain error: {e}")


# ═══════════════════════════════════════════════════════════════
# RL (REINFORCEMENT LEARNING) ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.route('/smart_suggest_rl', methods=['POST'])
def smart_suggest_rl():
    """
    RL-based smart suggestion using Q-Learning
    Returns suggestion based on learned Q-values
    """
    try:
        data = request.get_json()
        
        # Get features from request
        features = {
            'completion_rate': float(data.get('completion_rate', 0)),
            'consistency': float(data.get('consistency', 0)),
            'drop_rate': float(data.get('drop_rate', 0)),
            'active_streaks': float(data.get('activeStreaks', 0)),
            'best_streak': float(data.get('bestStreak', 0)),
            'total_habits': float(data.get('totalHabits', 0)),
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'is_weekend': datetime.now().weekday() >= 5,
            'previous_completion': float(data.get('previous_completion', 0)),
            'trend_7d': float(data.get('trend_7d', 0)),
            'trend_30d': float(data.get('trend_30d', 0))
        }
        
        # Get available suggestions
        habit_names = data.get('habit_names', [])
        available = detect_missing_categories(habit_names)
        if not available:
            available = SUGGESTIONS
        
        # Get state vector
        state = rl_engine.get_state_vector(features)
        
        # Select action using RL policy
        suggestion_id = rl_engine.select_action(state, available, explore=True)
        
        # Get Q-values for transparency
        q_values = rl_engine.get_q_values(state)
        top_q = sorted(q_values.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return jsonify({
            'suggestion_id': suggestion_id,
            'suggestion_text': get_suggestion_text(suggestion_id),
            'confidence': round(1.0 - rl_engine.epsilon, 3),
            'exploration': rl_engine.epsilon > 0.1,
            'method': 'rl_q_network',
            'top_q_values': [
                {'suggestion': k, 'q_value': round(v, 3)} for k, v in top_q
            ],
            'epsilon': round(rl_engine.epsilon, 3),
            'engine_stats': rl_engine.get_stats()
        })
        
    except Exception as e:
        print(f"❌ RL Suggestion Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/feedback', methods=['POST'])
def receive_feedback():
    """
    Receive user feedback for RL training
    Stores experience (state, action, reward, next_state)
    """
    try:
        data = request.get_json()
        
        # Required fields
        user_id = data.get('userId', 'anonymous')
        suggestion_id = data.get('suggestionId')
        interaction = data.get('interaction')  # 'completed', 'opened', 'ignored', 'deleted', 'snoozed'
        state_before = data.get('stateBefore', {})
        state_after = data.get('stateAfter', {})
        
        if not suggestion_id or not interaction:
            return jsonify({'error': 'Missing suggestionId or interaction'}), 400
        
        # Calculate reward
        reward_map = {
            'completed': 1.0,
            'opened': 0.3,
            'app_opened': 0.1,
            'ignored': -0.2,
            'deleted': -0.5,
            'snoozed': -0.1
        }
        reward = reward_map.get(interaction, 0.0)
        
        # Convert to state vectors
        state_vector = rl_engine.get_state_vector(state_before)
        next_state_vector = rl_engine.get_state_vector(state_after)
        
        # Store experience
        rl_engine.store_experience(
            state=state_vector,
            action=suggestion_id,
            reward=reward,
            next_state=next_state_vector,
            done=False
        )
        
        # Add to feedback buffer for batch processing
        rl_feedback_buffer.append({
            'user_id': user_id,
            'suggestion_id': suggestion_id,
            'reward': reward,
            'timestamp': datetime.now().isoformat()
        })
        
        # Trigger training if enough feedback
        global rl_last_training
        if len(rl_engine.replay_buffer) >= 32 and (time.time() - rl_last_training) > RL_TRAINING_INTERVAL:
            print(f"🧠 Auto-training RL: {len(rl_engine.replay_buffer)} experiences")
            loss = rl_engine.train_on_batch(num_steps=10)
            rl_last_training = time.time()
            
            # Save periodically
            if rl_engine.training_step % 100 == 0:
                rl_engine.save_model()
                print(f"💾 RL Model saved at step {rl_engine.training_step}")
            
            return jsonify({
                'success': True,
                'reward': reward,
                'training_triggered': True,
                'loss': round(loss, 6),
                'buffer_size': len(rl_engine.replay_buffer),
                'epsilon': round(rl_engine.epsilon, 3)
            })
        
        return jsonify({
            'success': True,
            'reward': reward,
            'training_triggered': False,
            'buffer_size': len(rl_engine.replay_buffer),
            'epsilon': round(rl_engine.epsilon, 3)
        })
        
    except Exception as e:
        print(f"❌ Feedback Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/rl_stats', methods=['GET'])
def rl_stats():
    """Get RL engine statistics"""
    try:
        stats = rl_engine.get_stats()
        stats['feedback_buffer_size'] = len(rl_feedback_buffer)
        stats['rl_last_training'] = rl_last_training
        stats['time_since_training'] = round(time.time() - rl_last_training, 0)
        stats['suggestions_list'] = SUGGESTIONS
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/rl_train', methods=['POST'])
def rl_manual_train():
    """Manually trigger RL training"""
    try:
        if len(rl_engine.replay_buffer) < 32:
            return jsonify({
                'error': 'Not enough experiences',
                'buffer_size': len(rl_engine.replay_buffer),
                'required': 32
            }), 400
        
        steps = request.json.get('steps', 50)
        losses = []
        
        for i in range(steps):
            loss = rl_engine.train_step()
            if loss is not None:
                losses.append(loss)
        
        avg_loss = np.mean(losses) if losses else 0.0
        
        # Update target network
        rl_engine.update_target_network()
        
        # Save model
        rl_engine.save_model()
        
        global rl_last_training
        rl_last_training = time.time()
        
        return jsonify({
            'success': True,
            'steps_trained': steps,
            'average_loss': round(avg_loss, 6),
            'final_epsilon': round(rl_engine.epsilon, 3),
            'training_step': rl_engine.training_step,
            'buffer_size': len(rl_engine.replay_buffer)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/rl_qvalues', methods=['POST'])
def rl_qvalues():
    """Get Q-values for a given state (for debugging)"""
    try:
        data = request.get_json()
        features = {
            'completion_rate': float(data.get('completion_rate', 0)),
            'consistency': float(data.get('consistency', 0)),
            'drop_rate': float(data.get('drop_rate', 0)),
            'active_streaks': float(data.get('activeStreaks', 0)),
            'best_streak': float(data.get('bestStreak', 0)),
            'total_habits': float(data.get('totalHabits', 0)),
            'hour': data.get('hour', datetime.now().hour),
            'day_of_week': data.get('day_of_week', datetime.now().weekday()),
            'is_weekend': data.get('is_weekend', datetime.now().weekday() >= 5),
            'previous_completion': float(data.get('previous_completion', 0)),
            'trend_7d': float(data.get('trend_7d', 0)),
            'trend_30d': float(data.get('trend_30d', 0))
        }
        
        state = rl_engine.get_state_vector(features)
        q_values = rl_engine.get_q_values(state)
        
        return jsonify({
            'features': features,
            'state_vector': state.tolist(),
            'q_values': q_values,
            'best_action': max(q_values.items(), key=lambda x: x[1]),
            'epsilon': rl_engine.epsilon
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if initialize_system():
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to initialize system. Exiting...")
        sys.exit(1)
