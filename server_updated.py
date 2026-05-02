from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from behavioral_ai_fixed_clean import BehavioralAIEngine
from train_model import HabitModelTrainer
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
# Enable CORS for all origins (development mode)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize components
behavioral_engine = None
model_trainer = None

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
        print("Behavioral AI system initialized successfully")
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

if __name__ == '__main__':
    if initialize_system():
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to initialize system. Exiting...")
        sys.exit(1)
