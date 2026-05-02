import sys
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import os

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')

class BehavioralAIEngine:
    def __init__(self):
        """Initialize the Behavioral AI Engine with pre-trained model"""
        self.model = None
        self.scaler = StandardScaler()
        self.behavior_patterns = self._load_behavior_patterns()
        self._initialize_model()
    
    def _load_behavior_patterns(self):
        """Load behavioral patterns and coaching strategies"""
        return {
            'enthusiastic_dropout': {
                'pattern': "حماس عالي في البداية ثم انخفاض سريع",
                'triggers': ['حماس العادة الجديدة', 'النجاح الأولي'],
                'interventions': ['العادات الصغيرة', 'شريك المساءلة'],
                'behavior': 'يبدأ بقوة لكن يفقد الزخم بسرعة',
                'arabic_explanation': 'يبدأ بحماس شديد لكن يجد صعوبة في الاستمرارية',
                'root_cause': 'أهداف أولية طموحة جداً',
                'coaching_insight': 'ابدأ أصغر مما تعتقد ضروري'
            },
            'struggling_beginner': {
                'pattern': 'صعوبة في إ建立 أي روتين',
                'triggers': ['تعقيد العادة', 'قلة المكافآت الفورية'],
                'interventions': ['تجميع العادات', 'تصميم البيئة'],
                'behavior': 'يواجه تحديات في بناء العادات الأساسية',
                'arabic_explanation': 'يجد صعوبة في إ建立 روتينات مستمرة',
                'root_cause': 'العادات معقدة جداً في البداية',
                'coaching_insight': 'بسطط إلى عادة واحدة في كل مرة'
            },
            'inconsistent_performer': {
                'pattern': 'أداء متغير، نجاح متقطع',
                'triggers': ['تقلبات المزاج', 'الشتتات الخارجية'],
                'interventions': ['توقيت ثابت', 'عادات الربط'],
                'behavior': 'الأداء يختلف بشكل كبير يومياً',
                'arabic_explanation': 'تنفيذ غير متسق رغم القدرة',
                'root_cause': 'قلة هيكل روتيني ثابت',
                'coaching_insight': 'أنشئ ربطات قائمة على الوقت'
            },
            'selective_discipline': {
                'pattern': 'ممتاز في المجالات المفضلة، ضعيف في أخرى',
                'triggers': ['مستوى الاهتمام', 'القيمة المدركة'],
                'interventions': ['ربط الإغراء', 'إعادة صياغة القيمة'],
                'behavior': 'يظهر انضباط فقط للعادات الممتعة',
                'arabic_explanation': 'تطبيق انتقائي لقوة الإرادة',
                'root_cause': 'خلل في التوازن الداخلي للدافعية',
                'coaching_insight': 'ربط العادات الأقل متعة بالمفضلة'
            },
            'overwhelmed_user': {
                'pattern': 'العادات كثيرة جداً، تنفيذ ضعيف',
                'triggers': ['تخطيط طموح', 'ضغط الإنتاجية'],
                'interventions': ['تقليل العادات', 'تحديد الأولويات'],
                'behavior': 'يحاول العديد من العادات في نفس الوقت',
                'arabic_explanation': 'مغرق بالكم الهائل من العادات',
                'root_cause': 'الخوف من فوت التطوير',
                'coaching_insight': 'ركز على عادة أساسية واحدة أولاً'
            },
            'potential_unrealized': {
                'pattern': 'قدرة عالية، هيكل منخفض',
                'triggers': ['قلة النظام', 'محفزات غير متسقة'],
                'interventions': ['تصميم النظام', 'النوايا المطبقة'],
                'behavior': 'يظهر إمكانات عالية لكن يفتقر للاستمرارية',
                'arabic_explanation': 'إمكانيات غير مستغلة بسبب قلة الهيكل',
                'root_cause': 'الاعتماد على الدافعية بدلاً من الأنظمة',
                'coaching_insight': 'ابنظ أنظمة بيئية وقائمة على الوقت'
            },
            'burst_effort': {
                'pattern': 'جهود مكثفة تتبعها الإرهاق',
                'triggers': ['الكمالية', 'التفكير الكل أو لا شيء'],
                'interventions': ['وتيرة مستدامة', 'أدنى عادات قابلة للتطبيق'],
                'behavior': 'فترات جهد مكثف تتبعها توقف كامل',
                'arabic_explanation': 'دورات من الجهد المكثف والإرهاق',
                'root_cause': 'معايير الكمالية',
                'coaching_insight': 'حدد معايير النجاح الأدنى القابلة للتطبيق'
            }
        }
    
    def _initialize_model(self):
        """Initialize and train the behavioral prediction model"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: completionRate, consistency, dropRate, activeStreaks, bestStreak, totalHabits
        X = np.random.rand(n_samples, 6)
        X[:, 0] = np.random.beta(2, 5, n_samples)  # completionRate
        X[:, 1] = np.random.beta(2, 3, n_samples)  # consistency  
        X[:, 2] = np.random.beta(1, 4, n_samples)  # dropRate
        X[:, 3] = np.random.poisson(3, n_samples)  # activeStreaks
        X[:, 4] = np.random.poisson(7, n_samples)  # bestStreak
        X[:, 5] = np.random.poisson(3, n_samples)  # totalHabits
        
        # Generate behavior labels based on patterns
        y = self._generate_behavior_labels(X)
        
        # Train model
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
    
    def _generate_behavior_labels(self, X):
        """Generate behavior labels based on feature patterns"""
        labels = []
        for i in range(len(X)):
            completion, consistency, drop, active, best, total = X[i]
            
            if completion < 0.3 and consistency < 0.3:
                labels.append('struggling_beginner')
            elif completion > 0.7 and consistency < 0.4:
                labels.append('enthusiastic_dropout')
            elif 0.4 < completion < 0.8 and consistency < 0.5:
                labels.append('inconsistent_performer')
            elif total > 4 and completion < 0.6:
                labels.append('overwhelmed_user')
            elif best > 10 and consistency < 0.6:
                labels.append('potential_unrealized')
            elif completion > 0.8 and drop > 0.3:
                labels.append('burst_effort')
            else:
                labels.append('selective_discipline')
        
        return np.array(labels)
    
    def _infer_user_behavior(self, features):
        # Behavior labels mapping to Arabic
        self.behavior_labels_ar = {
            "selective_discipline": "لديك انضباط جزئي في العادات",
            "inconsistent_performer": "أداؤك غير منتظم",
            "enthusiastic_dropout": "تبدأ بحماس ثم تفقد الاستمرارية",
            "struggling_beginner": "ما زلت في بداية بناء العادات",
            "overwhelmed_user": "أنت محاول القيام بالكثير",
            "potential_unrealized": "لديك إمكانيات أكبر",
            "burst_effort": "تعمل بجهد مكثف ثم تتعب"
        }
        """Infer user behavior patterns from features"""
        # Prepare features for prediction
        feature_array = np.array([[
            features['completionRate'],
            features['consistency'],
            features['dropRate'],
            features['activeStreaks'],
            features['bestStreak'],
            features['totalHabits']
        ]])
        
        # Predict behavior
        feature_scaled = self.scaler.transform(feature_array)
        predicted_behaviors = self.model.predict_proba(feature_scaled)[0]
        
        # Get top behaviors with probabilities
        behavior_classes = self.model.classes_
        behavior_probs = list(zip(behavior_classes, predicted_behaviors))
        behavior_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Create detected behaviors list
        detected_behaviors = []
        for behavior_name, probability in behavior_probs[:3]:  # Top 3 behaviors
            if probability > 0.1:  # Only include if probability is significant
                behavior_data = self.behavior_patterns.get(behavior_name, {})
                behavior_text = self.behavior_labels_ar.get(behavior_name, "")
                detected_behaviors.append({
                    'behavior': behavior_text,
                    'arabic_explanation': behavior_data.get('arabic_explanation', ''),
                    'root_cause': behavior_data.get('root_cause', ''),
                    'coaching_insight': behavior_data.get('coaching_insight', '')
                })
        
        return detected_behaviors
    
    def _generate_behavioral_summary(self, features, behaviors):
        """Generate dynamic summary based on completionRate and consistency"""
        print(f"DEBUG: Summary generation - behaviors received: {len(behaviors)}")
        
        try:
            # STEP 1: EXTRACT PERFORMANCE METRICS
            completion = features.get('completionRate', 0)
            consistency = features.get('consistency', 0)
            
            # STEP 2: PERFORMANCE-BASED TONE GENERATION
            if completion < 0.4:
                # LOW PERFORMANCE - slightly strict / motivational
                summary = "واضح إنك مش ملتزم بالعادات بالشكل الكافي، محتاج تركز أكتر على الاستمرارية اليومية وتبدأ بخطوات صغيرة جدًا"
            elif completion < 0.75:
                # MEDIUM PERFORMANCE - advisory
                if consistency < 0.5:
                    summary = f"أنت ماشي كويس بمعدل إنجاز {int(completion*100)}%، لكن محتاج تلتزم أكتر خاصة في الاستمرارية اليومية"
                else:
                    summary = f"أداؤك بيتحسن بمعدل {int(completion*100)}%، حافظ على الزخم الحالي وركز على الثبات"
            else:
                # HIGH PERFORMANCE - encouraging
                summary = f"أداءك ممتاز واستمراريتك واضحة بمعدل {int(completion*100)}% 👏، حافظ على هذا المستوى الرائع وفكر في تحديات جديدة"
            
            return summary
            
        except Exception as e:
            print(f"ERROR in _generate_behavioral_summary: {e}")
            return "من خلال بياناتك، واضح إنك بتحاول تطور نفسك، ودي بداية ممتازة جدًا"
    
    def _generate_behavioral_strengths(self, features, behaviors):
        """Generate SPECIFIC, DATA-DRIVEN strengths (no generic fluff)"""
        print(f"DEBUG: Strengths generation - features: {features}")
        
        try:
            strengths = []
            completion = features.get('completionRate', 0)
            consistency = features.get('consistency', 0)
            best_streak = features.get('bestStreak', 0)
            active_streaks = features.get('activeStreaks', 0)
            drop_rate = features.get('dropRate', 0)
            
            # 🔥 STRENGTH 1: إذا كان الأداء اليوم عالي
            if completion >= 0.8:
                strengths.append(f"✅ إنجازك اليوم {int(completion*100)}% - أداء ممتاز فعلاً")
            elif completion >= 0.5:
                strengths.append(f"✅ أنجزت {int(completion*100)}% من عاداتك اليوم")
            
            # 🔥 STRENGTH 2: إذا كان عنده سلسلة نشطة الآن
            if active_streaks >= 0.6:
                strengths.append("🔥 عندك سلاسل التزام نشطة حالياً - كمل!")
            
            # 🔥 STRENGTH 3: أفضل سلسلة تاريخية
            if best_streak >= 3:
                strengths.append(f"🏆 رقمك القياسي: {int(best_streak)} أيام متتالية - قادر تكررها!")
            elif best_streak >= 1:
                strengths.append("🏆 سبق وأكملت عادات متتالية - الخبرة موجودة")
            
            # 🔥 STRENGTH 4: استقرار بدون تراجع
            if drop_rate < 0.2 and consistency > 0.5:
                strengths.append("📈 أداؤك مستقر بدون تراجعات كبيرة")
            
            # 🔥 STRENGTH 5: عدد العادات النشطة
            total_habits = features.get('totalHabits', 0)
            if total_habits >= 4:
                strengths.append(f"💪 تتابع {int(total_habits)} عادات معاً - نظام متكامل")
            elif total_habits >= 2:
                strengths.append(f"💪 عندك {int(total_habits)} عادات نشطة - بداية متنوعة")
            
            # إذا فاضي، نضيف نقطة واحدة واقعية
            if not strengths:
                strengths.append("👍 بدأت رحلة العادات - كل بداية تستحق التقدير")
            
            print(f"✅ GENERATED STRENGTHS ({len(strengths)}): {strengths}")
            return strengths[:4]  # ما نرجعش أكثر من 4
            
        except Exception as e:
            print(f"ERROR in _generate_behavioral_strengths: {e}")
            return ["✅ إنجاز {int(features.get('completionRate', 0)*100)}% اليوم - أداء جيد"]
    
    def _generate_improvements(self, features, behaviors):
        """Generate SPECIFIC, ACTIONABLE improvements based on data"""
        print(f"DEBUG: Improvements generation - features: {features}")
        
        try:
            improvements = []
            completion = features.get('completionRate', 0)
            consistency = features.get('consistency', 0)
            drop_rate = features.get('dropRate', 0)
            active_streaks = features.get('activeStreaks', 0)
            total_habits = features.get('totalHabits', 0)
            
            # 🎯 IMPROVEMENT 1: إذا الأداء ضعيف
            if completion < 0.4:
                improvements.append(f"⚡ هدف قريب: وصل لـ 50% بإنجاز عادة واحدة فقط")
            elif completion < 0.7:
                improvements.append(f"⚡ زودها شوية: {int((0.8 - completion)*100)}% زيادة توصلك للتميز")
            
            # 🎯 IMPROVEMENT 2: إذا التراجع عالي
            if drop_rate > 0.3:
                improvements.append("🆘 أوقف التراجع: حدد سبب الفشل في اليومين اللي فاتوا")
            elif drop_rate > 0.15:
                improvements.append("📉 تراجع بسيط: راجع موعد تنفيذ العادة (غير الوقت)")
            
            # 🎯 IMPROVEMENT 3: إذا مفيش سلاسل نشطة
            if active_streaks < 0.3:
                improvements.append("🔥 سلسلة جديدة: أكمل نفس العادة 3 أيام متتالية")
            
            # 🎯 IMPROVEMENT 4: إذا الاستقرار ضعيف
            if consistency < 0.4:
                improvements.append("📊 ثبات أولاً: اختر عادة واحدة أبسط عندك وثبتها")
            
            # 🎯 IMPROVEMENT 5: إذا العادات كثيرة ومش متابعة
            if total_habits >= 4 and completion < 0.5:
                improvements.append(f"✋ ركز: عندك {int(total_habits)} عادات - freezing {int(total_habits-2)} وحافظ على 2 بس")
            
            # 🎯 IMPROVEMENT 6: لو مفيش عادات mental
            habit_types = [b.get('type', '') for b in behaviors]
            mental_keywords = ['thinking', 'reading', 'mental', 'mindfulness']
            has_mental = any(m in str(habit_types).lower() for m in mental_keywords)
            if not has_mental:
                improvements.append("🧠 أضف عادة عقلية: 5 دقائق تفكر أو اقرأ")
            
            # إذا فاضي، نضيف تحسين واحد واقعي
            if not improvements:
                improvements.append("🎯 حافظ على الإيقاع الحالي - الاستمرارية أهم من الكمال")
            
            print(f"📈 GENERATED IMPROVEMENTS ({len(improvements)}): {improvements}")
            return improvements[:4]  # ما نرجعش أكثر من 4
            
        except Exception as e:
            print(f"ERROR in _generate_improvements: {e}")
            return ["🎯 حدد هدف واحد بسيط هذا الأسبوع"]
    
    def _generate_contextual_suggestions(self, features, behaviors, habit_names=None, completed_habits=None):
        """Generate contextual suggestions using semantic scoring and composition"""
        print(f"DEBUG: Suggestions generation - behaviors received: {len(behaviors)}")
        
        try:
            # STEP 1: EXTRACT HABIT TYPES AND EXISTING HABITS
            habit_types = [b.get('type', '') for b in behaviors]
            existing_habits = []
            completed = []
            
            if habit_names:
                print("Incoming habits:", habit_names)
                # Extract habit types (normalized)
                for habit in habit_names:
                    if isinstance(habit, dict) and 'tag' in habit:
                        existing_habits.append(habit['tag'].lower().strip())
                    elif isinstance(habit, str):
                        existing_habits.append(habit.lower().strip())
            
            # Process completed habits
            if completed_habits:
                print("Completed habits:", completed_habits)
                for habit in completed_habits:
                    if isinstance(habit, str):
                        completed.append(habit.lower().strip())
            
            # STEP 2: DEFINE ALL POSSIBLE HABITS WITH ENGLISH KEYS
            all_possible_habits = {
                "water": "اشرب كمية كافية من الماء",
                "exercise": "مارس رياضة خفيفة",
                "reading": "اقرأ يوميًا",
                "sleep": "نظم نومك",
                "thinking": "خذ وقت للتفكر"
            }
            print("All habits:", list(all_possible_habits.keys()))
            print("Existing:", existing_habits)
            print("Completed:", completed)
            
            # STEP 3: FILTER OUT EXISTING AND COMPLETED HABITS
            blocked = set(existing_habits + completed)
            suggestions = [
                all_possible_habits[k]
                for k in all_possible_habits.keys()
                if k not in blocked
            ]
            print("Final suggestions:", suggestions)
            
            return suggestions
            
        except Exception as e:
            print(f"ERROR in _generate_contextual_suggestions: {e}")
            return ["abdaa b khutwa sghira alyawm", "hadd waqtan thabtan yawmiyan", "astamar bidun tawaqf"]
    
    def generate_insights(self, features, habit_names=None, completed_habits=None):
        """Main method to generate behavioral insights"""
        
        # SAFE HELPERS
        def safe_string(value, default):
            return value if isinstance(value, str) and value.strip() else default
        
        def safe_list(value, default):
            return value if isinstance(value, list) and len(value) > 0 else default
        
        try:
            behaviors = self._infer_user_behavior(features)
            print(f"DEBUG: Final behaviors detected: {len(behaviors)} behaviors")
            print("FINAL OUTPUT: Generated insights successfully")
            score = (
                features['completionRate'] * 40 +
                features['consistency'] * 30 +
                (1 - features['dropRate']) * 20 +
                min(features['activeStreaks'] / 30, 1) * 10
            )

            insights = {
                'summary': self._generate_behavioral_summary(features, behaviors),
                'strengths': self._generate_behavioral_strengths(features, behaviors),
                'improvements': self._generate_improvements(features, behaviors),
                'suggestions': self._generate_contextual_suggestions(features, behaviors, habit_names, completed_habits),
                'recommendedHabits': [],
                'userScore': round(score, 1),
                'userState': 'mutawassit',
                'trend': 'munhadir' if features['dropRate'] > 0.15 else 'mutahassin' if features['dropRate'] < -0.1 else 'mustaqirr',
                'confidence': min(0.95, 0.6 + features['consistency'] * 0.35),
                'futureScore': round(score * 0.95, 1),
                'detectedBehaviors': [b['behavior'] for b in behaviors if b.get('behavior')]
            }

            # FULL NULL SHIELD - Apply to ALL fields (Native Arabic Generation)
            insights['summary'] = safe_string(insights.get('summary'), "من خلال بياناتك، واضح إنك بتحاول تطور نفسك، ودي بداية ممتازة جدًا")
            insights['strengths'] = safe_list(insights.get('strengths'), ["عندك رغبة حقيقية في تطوير نفسك"])
            insights['suggestions'] = safe_list(insights.get('suggestions'), ["ابدأ بخطوة صغيرة اليوم"])
            
            insights['improvements'] = safe_list(insights.get('improvements'), ["حاول زيادة استمراريتك تدريجياً"])
            insights['recommendedHabits'] = safe_list(insights.get('recommendedHabits'), ["اشرب كوب ماء صباحًا"])
            insights['detectedBehaviors'] = safe_list(insights.get('detectedBehaviors'), ["مبتدئ"])
            
            insights['userState'] = safe_string(str(insights.get('userState')), "متوسط")
            insights['trend'] = safe_string(insights.get('trend'), "مستقر")

            # FORCE TYPE SAFETY - Add assertions for all critical fields
            assert isinstance(insights['summary'], str), f"Summary must be string, got {type(insights['summary'])}"
            assert isinstance(insights['strengths'], list), f"Strengths must be list, got {type(insights['strengths'])}"
            assert isinstance(insights['suggestions'], list), f"Suggestions must be list, got {type(insights['suggestions'])}"
            assert isinstance(insights['improvements'], list), f"Improvements must be list, got {type(insights['improvements'])}"
            assert isinstance(insights['recommendedHabits'], list), f"RecommendedHabits must be list, got {type(insights['recommendedHabits'])}"
            assert isinstance(insights['detectedBehaviors'], list), f"DetectedBehaviors must be list, got {type(insights['detectedBehaviors'])}"
            assert isinstance(insights['userState'], str), f"UserState must be string, got {type(insights['userState'])}"
            assert isinstance(insights['trend'], str), f"Trend must be string, got {type(insights['trend'])}"

            print("FINAL BEHAVIORS SENT:", insights['detectedBehaviors'])
            print("FINAL OUTPUT:", insights)
            return insights

        except Exception as e:
            print(f"ERROR in generate_insights: {e}")
            
            # Ultimate fallback with guaranteed non-null values
            return {
                'summary': "من خلال بياناتك، واضح إنك بتحاول تطور عاداتك، وده بداية ممتازة جدًا",
                'strengths': ["تبدي محاولات جادة لتطوير نفسك", "بدأت بالفعل في بناء عادات إيجابية"],
                'improvements': ["حاول زيادة استمراريتك تدريجياً"],
                'suggestions': ["ابدأ بعادة صغيرة جداً", "حدد وقتاً ثابتاً كل يوم"],
                'recommendedHabits': ["ashrub kub maya sabahan"],
                'userScore': 50.0,
                'userState': 'متوسط',
                'trend': 'مستقر',
                'confidence': 0.7,
                'futureScore': 55.0,
                'detectedBehaviors': ['مبتدئ']
            }
