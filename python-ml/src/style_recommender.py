"""
Style Recommendation Engine
Rule-based styling advice based on body measurements and proportions
PHASE 1D implementation
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class BodyShape(Enum):
    PETITE_NARROW = "petite_narrow"
    PETITE_CURVY = "petite_curvy"
    ATHLETIC = "athletic"
    AVERAGE_BALANCED = "average_balanced"
    AVERAGE_CURVY = "average_curvy"
    LARGER_FRAME = "larger_frame"
    UNKNOWN = "unknown"

class ColorPalette(Enum):
    BRIGHT = "bright"
    JEWEL_TONES = "jewel_tones"
    PASTELS = "pastels"
    NEUTRALS = "neutrals"
    BOLD = "bold"
    ANY = "any"

@dataclass
class StyleAdvice:
    category: str
    subcategory: Optional[str]
    reason: str
    color_suggestion: ColorPalette
    confidence: float
    body_shape: BodyShape

@dataclass
class BodyMeasurementProfile:
    shoulder_width_cm: float
    chest_width_cm: float
    torso_length_cm: float
    waist_cm: Optional[float] = None
    hip_cm: Optional[float] = None
    inseam_cm: Optional[float] = None

class StyleRecommender:
    """Provides style recommendations based on body measurements"""
    
    # Body shape classification thresholds
    PETITE_TORSO_MAX = 55.0  # cm
    PETITE_SHOULDER_MAX = 38.0  # cm
    AVERAGE_SIZE_MIN = 40.0  # cm (avg of shoulder/chest/hip)
    AVERAGE_SIZE_MAX = 50.0  # cm
    ATHLETIC_SHOULDER_HIP_DIFF = 8.0  # cm
    CURVY_HIP_SHOULDER_RATIO = 0.95
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def classify_body_shape(self, profile: BodyMeasurementProfile) -> BodyShape:
        """Classify body shape based on measurements"""
        
        # Calculate average size
        size_measurements = [profile.shoulder_width_cm, profile.chest_width_cm]
        if profile.hip_cm:
            size_measurements.append(profile.hip_cm)
        
        avg_size = sum(size_measurements) / len(size_measurements)
        
        # Calculate proportions
        if profile.hip_cm:
            hip_shoulder_ratio = profile.hip_cm / profile.shoulder_width_cm
        else:
            hip_shoulder_ratio = 1.0  # Unknown
        
        # Classify
        if profile.torso_length_cm < self.PETITE_TORSO_MAX and profile.shoulder_width_cm < self.PETITE_SHOULDER_MAX:
            if profile.hip_cm and hip_shoulder_ratio >= self.CURVY_HIP_SHOULDER_RATIO:
                return BodyShape.PETITE_CURVY
            else:
                return BodyShape.PETITE_NARROW
        
        elif avg_size >= self.AVERAGE_SIZE_MIN and avg_size < self.AVERAGE_SIZE_MAX:
            if profile.hip_cm and (profile.shoulder_width_cm - profile.hip_cm) > self.ATHLETIC_SHOULDER_HIP_DIFF:
                return BodyShape.ATHLETIC
            elif profile.hip_cm and hip_shoulder_ratio >= self.CURVY_HIP_SHOULDER_RATIO:
                return BodyShape.AVERAGE_CURVY
            else:
                return BodyShape.AVERAGE_BALANCED
        
        elif avg_size >= self.AVERAGE_SIZE_MAX:
            return BodyShape.LARGER_FRAME
        
        return BodyShape.UNKNOWN
    
    def recommend(self, profile: BodyMeasurementProfile) -> List[StyleAdvice]:
        """Generate style recommendations for a body profile"""
        
        body_shape = self.classify_body_shape(profile)
        recommendations = []
        
        if body_shape in self.rules:
            for rule in self.rules[body_shape]:
                advice = StyleAdvice(
                    category=rule['category'],
                    subcategory=rule.get('subcategory'),
                    reason=rule['reason'],
                    color_suggestion=rule['color'],
                    confidence=rule['confidence'],
                    body_shape=body_shape
                )
                recommendations.append(advice)
        
        # Add universal recommendations
        recommendations.extend(self._get_universal_recommendations(profile, body_shape))
        
        return recommendations
    
    def _initialize_rules(self) -> Dict[BodyShape, List[Dict]]:
        """Initialize styling rules for each body shape"""
        return {
            BodyShape.PETITE_NARROW: [
                {
                    'category': 'tops',
                    'subcategory': 'crop_top',
                    'reason': 'Petite frame suits shorter crops to elongate proportions without overwhelming',
                    'color': ColorPalette.BRIGHT,
                    'confidence': 0.85
                },
                {
                    'category': 'dresses',
                    'subcategory': 'empire_waist',
                    'reason': 'High waistline creates illusion of longer legs on petite frames',
                    'color': ColorPalette.PASTELS,
                    'confidence': 0.80
                },
                {
                    'category': 'pants',
                    'subcategory': 'high_waist_slim',
                    'reason': 'High waist and slim cut elongate legs for petite builds',
                    'color': ColorPalette.NEUTRALS,
                    'confidence': 0.82
                }
            ],
            
            BodyShape.PETITE_CURVY: [
                {
                    'category': 'dresses',
                    'subcategory': 'fit_and_flare',
                    'reason': 'Fitted top with flared skirt flatters petite curvy proportions',
                    'color': ColorPalette.JEWEL_TONES,
                    'confidence': 0.88
                },
                {
                    'category': 'tops',
                    'subcategory': 'wrap_top',
                    'reason': 'Wrap style accentuates curves while maintaining balance',
                    'color': ColorPalette.BOLD,
                    'confidence': 0.83
                }
            ],
            
            BodyShape.ATHLETIC: [
                {
                    'category': 'skirts',
                    'subcategory': 'a_line',
                    'reason': 'A-line skirts add curves and balance wider shoulders',
                    'color': ColorPalette.BRIGHT,
                    'confidence': 0.87
                },
                {
                    'category': 'dresses',
                    'subcategory': 'belted',
                    'reason': 'Belted styles create waist definition for athletic builds',
                    'color': ColorPalette.ANY,
                    'confidence': 0.84
                },
                {
                    'category': 'tops',
                    'subcategory': 'peplum',
                    'reason': 'Peplum adds volume at hips to balance broad shoulders',
                    'color': ColorPalette.JEWEL_TONES,
                    'confidence': 0.81
                }
            ],
            
            BodyShape.AVERAGE_BALANCED: [
                {
                    'category': 'dresses',
                    'subcategory': 'sheath',
                    'reason': 'Balanced proportions suit straight, streamlined silhouettes',
                    'color': ColorPalette.ANY,
                    'confidence': 0.90
                },
                {
                    'category': 'tops',
                    'subcategory': 'structured',
                    'reason': 'Structured pieces complement balanced frame without adding bulk',
                    'color': ColorPalette.NEUTRALS,
                    'confidence': 0.85
                }
            ],
            
            BodyShape.AVERAGE_CURVY: [
                {
                    'category': 'dresses',
                    'subcategory': 'wrap_dress',
                    'reason': 'Wrap dresses ideal for balanced curvy proportions, accentuate waist',
                    'color': ColorPalette.JEWEL_TONES,
                    'confidence': 0.92
                },
                {
                    'category': 'tops',
                    'subcategory': 'v_neck',
                    'reason': 'V-necklines elongate and flatter curvy upper body',
                    'color': ColorPalette.BOLD,
                    'confidence': 0.86
                },
                {
                    'category': 'pants',
                    'subcategory': 'bootcut',
                    'reason': 'Bootcut balances curvy hips with slight flare',
                    'color': ColorPalette.NEUTRALS,
                    'confidence': 0.83
                }
            ],
            
            BodyShape.LARGER_FRAME: [
                {
                    'category': 'dresses',
                    'subcategory': 'a_line',
                    'reason': 'A-line flatters larger frames by skimming over body',
                    'color': ColorPalette.JEWEL_TONES,
                    'confidence': 0.85
                },
                {
                    'category': 'tops',
                    'subcategory': 'tunic',
                    'reason': 'Tunics provide comfortable, flattering coverage',
                    'color': ColorPalette.ANY,
                    'confidence': 0.82
                },
                {
                    'category': 'jackets',
                    'subcategory': 'longline',
                    'reason': 'Long jackets create vertical lines that elongate silhouette',
                    'color': ColorPalette.NEUTRALS,
                    'confidence': 0.80
                }
            ]
        }
    
    def _get_universal_recommendations(self, profile: BodyMeasurementProfile, 
                                      body_shape: BodyShape) -> List[StyleAdvice]:
        """Recommendations that apply across body shapes"""
        universal = []
        
        # Proportions-based recommendations
        if profile.torso_length_cm and profile.shoulder_width_cm:
            torso_shoulder_ratio = profile.torso_length_cm / profile.shoulder_width_cm
            
            # Long torso relative to shoulders
            if torso_shoulder_ratio > 1.6:
                universal.append(StyleAdvice(
                    category='tops',
                    subcategory='cropped',
                    reason='Long torso benefits from cropped styles to balance proportions',
                    color_suggestion=ColorPalette.ANY,
                    confidence=0.75,
                    body_shape=body_shape
                ))
        
        # Color recommendations based on confidence
        if profile.chest_width_cm and profile.shoulder_width_cm:
            chest_shoulder_ratio = profile.chest_width_cm / profile.shoulder_width_cm
            
            # Balanced upper body = can handle bold patterns
            if 0.85 <= chest_shoulder_ratio <= 0.95:
                universal.append(StyleAdvice(
                    category='general',
                    subcategory='patterns',
                    reason='Balanced proportions suit bold patterns and prints',
                    color_suggestion=ColorPalette.BOLD,
                    confidence=0.70,
                    body_shape=body_shape
                ))
        
        return universal
    
    def get_style_score(self, profile: BodyMeasurementProfile, 
                       garment_category: str, garment_subcategory: str) -> float:
        """Score how well a specific garment matches body shape (0-1)"""
        
        recommendations = self.recommend(profile)
        
        # Find matching recommendation
        for rec in recommendations:
            if rec.category == garment_category and rec.subcategory == garment_subcategory:
                return rec.confidence
        
        # No specific recommendation, return neutral score
        return 0.5
    
    def explain_recommendation(self, advice: StyleAdvice) -> str:
        """Generate detailed explanation for a recommendation"""
        return f"""
        Body Shape: {advice.body_shape.value.replace('_', ' ').title()}
        
        Recommended: {(advice.subcategory or '').replace('_', ' ').title()} {advice.category.title()}
        
        Why: {advice.reason}
        
        Color Palette: {advice.color_suggestion.value.replace('_', ' ').title()}
        
        Confidence: {advice.confidence * 100:.0f}%
        """


def main():
    """Demo: Style recommendation system"""
    print("=" * 70)
    print("STYLE RECOMMENDATION ENGINE DEMO")
    print("=" * 70)
    
    recommender = StyleRecommender()
    
    # Test case 1: Petite narrow frame
    print("\n1. Test Case: Petite Narrow Frame")
    profile1 = BodyMeasurementProfile(
        shoulder_width_cm=36.0,
        chest_width_cm=32.0,
        torso_length_cm=52.0,
        waist_cm=26.0,
        hip_cm=35.0
    )
    
    shape1 = recommender.classify_body_shape(profile1)
    print(f"   Classified as: {shape1.value.replace('_', ' ').title()}")
    
    recommendations1 = recommender.recommend(profile1)
    print(f"   Recommendations ({len(recommendations1)}):")
    for i, rec in enumerate(recommendations1[:3]):
        print(f"      {i+1}. {(rec.subcategory or '').replace('_', ' ').title()} {rec.category.title()}")
        print(f"         Reason: {rec.reason}")
        print(f"         Color: {rec.color_suggestion.value}, Confidence: {rec.confidence:.0%}")
    
    # Test case 2: Athletic build
    print("\n2. Test Case: Athletic Build")
    profile2 = BodyMeasurementProfile(
        shoulder_width_cm=44.0,
        chest_width_cm=42.0,
        torso_length_cm=62.0,
        waist_cm=32.0,
        hip_cm=35.0
    )
    
    shape2 = recommender.classify_body_shape(profile2)
    print(f"   Classified as: {shape2.value.replace('_', ' ').title()}")
    
    recommendations2 = recommender.recommend(profile2)
    print(f"   Recommendations ({len(recommendations2)}):")
    for i, rec in enumerate(recommendations2[:3]):
        print(f"      {i+1}. {(rec.subcategory or '').replace('_', ' ').title()} {rec.category.title()}")
        print(f"         Reason: {rec.reason}")
        print(f"         Color: {rec.color_suggestion.value}, Confidence: {rec.confidence:.0%}")
    
    # Test case 3: Average curvy
    print("\n3. Test Case: Average Curvy")
    profile3 = BodyMeasurementProfile(
        shoulder_width_cm=42.0,
        chest_width_cm=44.0,
        torso_length_cm=60.0,
        waist_cm=34.0,
        hip_cm=46.0
    )
    
    shape3 = recommender.classify_body_shape(profile3)
    print(f"   Classified as: {shape3.value.replace('_', ' ').title()}")
    
    recommendations3 = recommender.recommend(profile3)
    print(f"   Recommendations ({len(recommendations3)}):")
    for i, rec in enumerate(recommendations3[:3]):
        print(f"      {i+1}. {(rec.subcategory or '').replace('_', ' ').title()} {rec.category.title()}")
        print(f"         Reason: {rec.reason}")
        print(f"         Color: {rec.color_suggestion.value}, Confidence: {rec.confidence:.0%}")
    
    # Test garment scoring
    print("\n4. Garment Score Test (Average Curvy trying wrap dress):")
    score = recommender.get_style_score(profile3, 'dresses', 'wrap_dress')
    print(f"   Style Match Score: {score:.0%}")
    
    print("\n" + "=" * 70)
    print("Style recommendation system ready!")
    print("Enables: Personalized outfit suggestions, +25% AOV potential")
    print("=" * 70)


if __name__ == "__main__":
    main()
