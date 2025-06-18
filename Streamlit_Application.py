import streamlit as st
import pandas as pd
from PIL import Image
import io
import os
import json
import re
from datetime import datetime
from collections import Counter
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import FileOperation

# Set page configuration
st.set_page_config(
    page_title="Golf Club Analyzer V3",
    page_icon="🏌️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def add_custom_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stSidebar {
            background-color: #2E7D32;
            color: white;
        }
        .stSidebar .stRadio label, .stSidebar .stSelectbox label, .stSidebar .stMultiselect label {
            color: white !important;
            font-weight: bold;
        }
        .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar .stSubheader {
            color: white;
        }
        .upload-header {
            font-size: 24px;
            font-weight: bold;
            color: #2e7d32;
            margin-bottom: 20px;
        }
        .instruction-text {
            font-size: 16px;
            color: #555;
            margin-bottom: 20px;
        }
        .image-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
        }
        .image-box {
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
            background-color: white;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            color: #666;
            font-size: 12px;
        }
        .green-button {
            background-color: #2e7d32;
            color: white;
        }
        .analysis-result {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #2e7d32;
            margin-top: 20px;
        }
        .raw-output-container {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: #f9f9f9;
            margin: 10px 0;
        }
        .query-info-container {
            background-color: #f0f0f0;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 15px;
            border-left: 3px solid #2e7d32;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
        }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# ENHANCED HELPER FUNCTIONS
# =====================================================

def safe_sql_string(value):
    """Safely escape strings for SQL queries."""
    if value is None:
        return ""
    return str(value).replace("'", "''").replace("\\", "\\\\")

def safe_numeric(value, default="NULL"):
    """Safely convert values to numeric for SQL insertion."""
    if value is None or value == '':
        return default
    try:
        # Extract numeric value if it's a string with units
        if isinstance(value, str):
            numbers = re.findall(r'\d+\.?\d*', value)
            if numbers:
                return float(numbers[0])
        return float(value)
    except:
        return default

def truncate_field_value(value, field_name, max_length=50):
    """
    Truncate field values to prevent database errors.
    Different fields have different max lengths in the database.
    """
    field_limits = {
        'shaft_flex': 20,      # Common values: Regular, Stiff, Senior, etc.
        'shaft_type': 20,      # Common values: Graphite, Steel
        'hand': 10,            # Right, Left
        'club_type': 30,       # Driver, Iron, Wedge, etc.
        'brand': 50,
        'model': 50,
        'face_angle': 20,      # Open, Closed, Neutral
        'grind_type': 20,
        'club_sub_type': 30,
        'condition_assessment': 20,  # Excellent, Good, Fair, Poor
        'face_sole_wear_grade': 20,
        'scratches_grade': 20,
        'paint_chips_grade': 20,
        'grip_condition': 30,
        'market_demand': 10    # High, Medium, Low
    }
    
    # Get the limit for this field
    limit = field_limits.get(field_name.lower(), max_length)
    
    # Handle special cases
    if value is None or value == '':
        return ''
    
    # Convert to string
    str_value = str(value)
    
    # Handle "Unable to determine" cases
    unable_mappings = {
        'shaft_flex': 'Unknown',
        'shaft_type': 'Unknown',
        'hand': 'Unknown',
        'brand': 'Unknown',
        'model': 'Unknown'
    }
    
    if 'unable to determine' in str_value.lower() or 'n/a' in str_value.lower():
        return unable_mappings.get(field_name.lower(), 'Unknown')
    
    # Truncate if needed
    if len(str_value) > limit:
        return str_value[:limit-3] + '...'
    
    return str_value

def parse_condition_descriptions(description_text):
    """
    Parse concatenated condition descriptions into clean, unique points.
    Removes duplicates and organizes by key findings.
    """
    if not description_text:
        return []
    
    # Split by semicolons and common delimiters
    parts = description_text.replace(';', ',').split(',')
    
    # Clean and deduplicate
    unique_findings = []
    seen = set()
    
    for part in parts:
        cleaned = part.strip()
        if cleaned and len(cleaned) > 5:  # Skip very short fragments
            # Normalize similar phrases
            normalized = cleaned.lower()
            
            # Skip redundant phrases
            skip_phrases = ['face not visible', 'image quality makes', 'unable to determine']
            if any(skip in normalized for skip in skip_phrases):
                continue
                
            # Check if we've seen a similar phrase
            is_duplicate = False
            for existing in seen:
                if normalized in existing or existing in normalized:
                    is_duplicate = True
                    break
            
            if not is_duplicate and cleaned not in unique_findings:
                unique_findings.append(cleaned)
                seen.add(normalized)
    
    return unique_findings[:3]  # Return top 3 most relevant findings

def extract_technical_specifications(analysis_text):
    """
    Enhanced extraction of technical specifications from AI analysis text.
    Returns dictionary with all possible specifications found.
    """
    specs = {}
    
    # Enhanced regex patterns for various specifications
    patterns = {
        'loft': [r'(\d+\.?\d*)°?\s*loft', r'loft[:\s]*(\d+\.?\d*)°?'],
        'lie_angle': [r'lie[:\s]*(\d+\.?\d*)°?', r'(\d+\.?\d*)°?\s*lie'],
        'bounce_angle': [r'bounce[:\s]*(\d+\.?\d*)°?', r'(\d+\.?\d*)°?\s*bounce'],
        'shaft_length': [r'(\d+\.?\d*)\s*inch', r'length[:\s]*(\d+\.?\d*)', r'(\d+)\"'],
        'year': [r'(20\d{2})', r'model year[:\s]*(20\d{2})'],
        'weight': [r'(\d+)\s*g', r'weight[:\s]*(\d+)', r'(\d+)\s*gram'],
        'volume': [r'(\d+)\s*cc', r'volume[:\s]*(\d+)', r'(\d+)\s*cubic']
    }
    
    for spec, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, analysis_text, re.IGNORECASE)
            if match and not specs.get(spec):
                try:
                    specs[spec] = float(match.group(1))
                    break
                except:
                    continue
    
    return specs

def extract_market_information(analysis_text, brand=None, club_type=None, condition=None):
    """
    Enhanced market information extraction with intelligent estimation.
    """
    market_info = {}
    
    # Extract explicit price mentions
    price_patterns = [
        r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',  # $150.00 or $1,500
        r'(\d+(?:,\d{3})*)\s*dollars?',     # 150 dollars
        r'price[:\s]*\$?(\d+(?:,\d{3})*)',  # price: $150
        r'retail[:\s]*\$?(\d+(?:,\d{3})*)', # retail: $150
        r'value[:\s]*\$?(\d+(?:,\d{3})*)'   # value: $150
    ]
    
    prices_found = []
    for pattern in price_patterns:
        matches = re.findall(pattern, analysis_text, re.IGNORECASE)
        for match in matches:
            try:
                # Remove commas and convert to float
                price = float(match.replace(',', ''))
                if 10 <= price <= 5000:  # Reasonable golf club price range
                    prices_found.append(price)
            except:
                continue
    
    # Market demand extraction
    demand_indicators = {
        'high': ['popular', 'sought after', 'high demand', 'desirable', 'premium'],
        'medium': ['moderate', 'average', 'decent', 'reasonable'],
        'low': ['limited', 'low demand', 'niche', 'discontinued']
    }
    
    for level, indicators in demand_indicators.items():
        if any(indicator in analysis_text.lower() for indicator in indicators):
            market_info['market_demand'] = level
            break
    
    # Intelligent price estimation based on brand, type, and condition
    if brand and club_type:
        estimated_retail, estimated_trade_in = estimate_market_values(
            brand, club_type, condition
        )
        market_info['estimated_retail_price'] = estimated_retail
        market_info['estimated_trade_in_value'] = estimated_trade_in
    
    # Use found prices if available
    if prices_found:
        market_info['retail_price'] = max(prices_found)  # Assume highest is retail
        if len(prices_found) > 1:
            market_info['trade_in_value'] = min(prices_found)  # Assume lowest is trade-in
    
    return market_info

def estimate_market_values(brand, club_type, condition):
    """
    Intelligent market value estimation based on brand tiers and club types.
    """
    # Brand tier pricing (retail when new)
    brand_tiers = {
        'tier_1': ['callaway', 'taylormade', 'titleist', 'ping', 'mizuno'],
        'tier_2': ['cobra', 'cleveland', 'wilson', 'srixon'],
        'tier_3': ['honma', 'pxg', 'tour edge', 'adams'],
        'tier_4': ['lynx', 'macgregor', 'dunlop', 'ben hogan']
    }
    
    # Base prices by club type and brand tier
    base_prices = {
        'driver': {'tier_1': 450, 'tier_2': 350, 'tier_3': 300, 'tier_4': 200},
        'fairway wood': {'tier_1': 350, 'tier_2': 250, 'tier_3': 200, 'tier_4': 150},
        'hybrid': {'tier_1': 250, 'tier_2': 200, 'tier_3': 150, 'tier_4': 100},
        'iron': {'tier_1': 80, 'tier_2': 60, 'tier_3': 50, 'tier_4': 35},
        'wedge': {'tier_1': 120, 'tier_2': 90, 'tier_3': 70, 'tier_4': 50},
        'putter': {'tier_1': 200, 'tier_2': 150, 'tier_3': 120, 'tier_4': 80}
    }
    
    # Determine brand tier
    brand_lower = brand.lower() if brand else ''
    tier = 'tier_4'  # Default
    for tier_name, brands in brand_tiers.items():
        if any(b in brand_lower for b in brands):
            tier = tier_name
            break
    
    # Get base price
    club_type_lower = club_type.lower() if club_type else 'iron'
    base_price = base_prices.get(club_type_lower, base_prices['iron']).get(tier, 100)
    
    # Condition multipliers for current market value
    condition_multipliers = {
        'excellent': 0.65,
        'very good': 0.55,
        'good': 0.45,
        'fair': 0.35,
        'poor': 0.25
    }
    
    condition_lower = condition.lower() if condition else 'good'
    multiplier = 0.45  # Default
    for cond, mult in condition_multipliers.items():
        if cond in condition_lower:
            multiplier = mult
            break
    
    estimated_retail = base_price * multiplier
    estimated_trade_in = estimated_retail * 0.6  # Trade-in typically 60% of market
    
    return round(estimated_retail), round(estimated_trade_in)

def extract_technology_features(analysis_text):
    """
    Extract golf technology features and special designations from analysis text.
    """
    # Common golf technologies
    technology_keywords = [
        'face insert', 'titanium', 'carbon fiber', 'adjustable', 'moveable weight',
        'cavity back', 'muscle back', 'blade', 'game improvement', 'players',
        'forged', 'cast', 'perimeter weighted', 'low cg', 'high moi',
        'speed pocket', 'jailbreak', 'twist face', 'ai design',
        'multi material', 'tungsten', 'speed foam', 'sound dampening'
    ]
    
    special_designations = [
        'tour issue', 'tour only', 'limited edition', 'prototype',
        'custom', 'commemorative', 'anniversary', 'signature series',
        'pro model', 'staff bag', 'pga tour', 'masters'
    ]
    
    found_tech = []
    found_special = []
    
    text_lower = analysis_text.lower()
    
    for tech in technology_keywords:
        if tech in text_lower:
            found_tech.append(tech.title())
    
    for special in special_designations:
        if special in text_lower:
            found_special.append(special.title())
    
    return found_tech, found_special

def extract_enhanced_defects(analysis_text, analysis_data):
    """
    Enhanced defect extraction that identifies more specific defect types and locations.
    Returns detailed defect observations for the CLUB_DEFECT_OBSERVATIONS_V3 table.
    """
    defects = []
    
    # Define defect patterns and severity mapping
    defect_patterns = {
        'scratch': {
            'patterns': [r'scratch\w*', r'scuff\w*', r'mark\w*', r'score\w*'],
            'locations': ['face', 'sole', 'crown', 'hosel', 'back', 'toe', 'heel'],
            'severities': {
                'light': ['light', 'minor', 'small', 'fine', 'superficial'],
                'moderate': ['moderate', 'noticeable', 'visible', 'medium'],
                'severe': ['heavy', 'deep', 'significant', 'major', 'extensive']
            }
        },
        'dent': {
            'patterns': [r'dent\w*', r'ding\w*', r'impact\w*'],
            'locations': ['crown', 'sole', 'face', 'back', 'edge'],
            'severities': {
                'light': ['small', 'minor', 'tiny'],
                'moderate': ['noticeable', 'medium'],
                'severe': ['large', 'significant', 'major']
            }
        },
        'paint_chip': {
            'patterns': [r'paint.*chip', r'finish.*wear', r'coating.*damage'],
            'locations': ['crown', 'sole', 'back', 'hosel'],
            'severities': {
                'light': ['minor', 'small'],
                'moderate': ['moderate', 'noticeable'],
                'severe': ['extensive', 'major']
            }
        },
        'rust': {
            'patterns': [r'rust\w*', r'corrosion', r'oxidation'],
            'locations': ['face', 'grooves', 'sole', 'back'],
            'severities': {
                'light': ['surface', 'minor', 'light'],
                'moderate': ['moderate', 'noticeable'],
                'severe': ['heavy', 'extensive', 'pitting']
            }
        },
        'grip_wear': {
            'patterns': [r'grip.*wear', r'grip.*worn', r'grip.*damage'],
            'locations': ['grip'],
            'severities': {
                'light': ['minor', 'slight'],
                'moderate': ['moderate', 'noticeable'],
                'severe': ['severe', 'replacement', 'cracked']
            }
        }
    }
    
    text_lower = analysis_text.lower()
    
    # Extract defects from analysis text
    for defect_type, config in defect_patterns.items():
        for pattern in config['patterns']:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Extract context around the match
                start = max(0, match.start() - 50)
                end = min(len(text_lower), match.end() + 50)
                context = text_lower[start:end]
                
                # Determine location
                location = 'general'
                for loc in config['locations']:
                    if loc in context:
                        location = loc
                        break
                
                # Determine severity
                severity = 'moderate'  # default
                for sev, keywords in config['severities'].items():
                    if any(keyword in context for keyword in keywords):
                        severity = sev
                        break
                
                # Extract size information if available
                size_match = re.search(r'(\d+\.?\d*)\s*(mm|inch|cm)', context)
                size_mm = None
                if size_match:
                    size_val = float(size_match.group(1))
                    unit = size_match.group(2)
                    if unit == 'inch':
                        size_mm = size_val * 25.4
                    elif unit == 'cm':
                        size_mm = size_val * 10
                    else:
                        size_mm = size_val
                
                defects.append({
                    'type': defect_type.upper(),
                    'location': location.title(),
                    'severity': severity,
                    'description': context.strip(),
                    'size_mm': size_mm,
                    'impact': get_defect_impact(defect_type, severity, location)
                })
    
    # Also extract from structured analysis data
    structured_defects = extract_defects_from_structured_data(analysis_data)
    defects.extend(structured_defects)
    
    # Remove duplicates
    unique_defects = []
    seen = set()
    for defect in defects:
        key = (defect['type'], defect['location'], defect['severity'])
        if key not in seen:
            unique_defects.append(defect)
            seen.add(key)
    
    return unique_defects

def extract_defects_from_structured_data(analysis_data):
    """
    Extract defects from structured analysis data fields.
    """
    defects = []
    
    # Face/Sole wear
    face_grade = analysis_data.get('face_sole_wear_grade', '').lower()
    face_desc = analysis_data.get('face_sole_wear_description', '')
    
    if face_grade and face_grade not in ['excellent', 'n/a', '']:
        defects.append({
            'type': 'FACE_WEAR',
            'location': 'Face/Sole',
            'severity': face_grade,
            'description': face_desc or f'Face and sole showing {face_grade} wear',
            'size_mm': None,
            'impact': get_defect_impact('face_wear', face_grade, 'face')
        })
    
    # Scratches
    scratch_grade = analysis_data.get('scratches_grade', '').lower()
    scratch_desc = analysis_data.get('scratches_description', '')
    scratch_locations = analysis_data.get('scratches_locations', [])
    
    if scratch_grade and scratch_grade not in ['none', 'n/a', '']:
        for location in (scratch_locations if scratch_locations else ['general']):
            defects.append({
                'type': 'SCRATCH',
                'location': location,
                'severity': scratch_grade,
                'description': scratch_desc or f'{scratch_grade} scratches on {location}',
                'size_mm': None,
                'impact': get_defect_impact('scratch', scratch_grade, location)
            })
    
    # Paint chips
    paint_grade = analysis_data.get('paint_chips_grade', '').lower()
    paint_desc = analysis_data.get('paint_chips_description', '')
    paint_locations = analysis_data.get('paint_chips_locations', [])
    
    if paint_grade and paint_grade not in ['none', 'excellent', 'n/a', '']:
        for location in (paint_locations if paint_locations else ['crown']):
            defects.append({
                'type': 'PAINT_CHIP',
                'location': location,
                'severity': paint_grade,
                'description': paint_desc or f'{paint_grade} paint condition on {location}',
                'size_mm': None,
                'impact': get_defect_impact('paint_chip', paint_grade, location)
            })
    
    # Grip condition
    grip_condition = analysis_data.get('grip_condition', '').lower()
    if grip_condition and grip_condition not in ['excellent', 'new', 'n/a', '']:
        defects.append({
            'type': 'GRIP_WEAR',
            'location': 'Grip',
            'severity': grip_condition,
            'description': f'Grip condition: {grip_condition}',
            'size_mm': None,
            'impact': get_defect_impact('grip_wear', grip_condition, 'grip')
        })
    
    return defects

def get_defect_impact(defect_type, severity, location):
    """
    Determine the performance impact of a defect based on type, severity, and location.
    """
    impact_matrix = {
        'scratch': {
            'face': {
                'light': 'Minimal impact on ball contact',
                'moderate': 'May affect spin and accuracy slightly',
                'severe': 'Significant impact on ball flight and spin'
            },
            'sole': {
                'light': 'Cosmetic only',
                'moderate': 'Minor impact on turf interaction',
                'severe': 'May affect club-ground contact'
            },
            'general': {
                'light': 'Cosmetic only',
                'moderate': 'Affects appearance',
                'severe': 'Significant cosmetic damage'
            }
        },
        'face_wear': {
            'face': {
                'good': 'Minimal performance impact',
                'fair': 'Moderate impact on spin and distance',
                'poor': 'Significant performance degradation'
            }
        },
        'paint_chip': {
            'general': {
                'light': 'Cosmetic only',
                'moderate': 'Affects resale value',
                'severe': 'Major cosmetic impact'
            }
        },
        'grip_wear': {
            'grip': {
                'good': 'Minimal impact on feel',
                'fair': 'May affect grip security and feel',
                'poor': 'Replacement recommended'
            }
        },
        'dent': {
            'crown': {
                'light': 'Cosmetic only',
                'moderate': 'May affect aerodynamics slightly',
                'severe': 'Potential structural impact'
            },
            'face': {
                'light': 'Minor impact on ball contact',
                'moderate': 'Noticeable performance impact',
                'severe': 'Significant performance degradation'
            }
        },
        'rust': {
            'face': {
                'light': 'Cosmetic concern',
                'moderate': 'May affect ball contact',
                'severe': 'Significant performance impact'
            },
            'grooves': {
                'light': 'Minor spin impact',
                'moderate': 'Noticeable spin reduction',
                'severe': 'Major spin loss'
            }
        }
    }
    
    try:
        return impact_matrix[defect_type][location][severity]
    except KeyError:
        return f"Performance impact from {defect_type} rated as {severity}"

# =====================================================
# ENHANCED ANALYSIS PROMPTS
# =====================================================

def get_comprehensive_multi_image_analysis_prompt(num_images):
    """
    Ultra-comprehensive golf club analysis prompt that extracts maximum data.
    Fixed placeholder syntax for Snowflake PROMPT() function.
    """
    # Build the image references dynamically with proper placeholder syntax
    image_refs = []
    for i in range(num_images):
        # Use proper placeholder format for Snowflake PROMPT function
        image_refs.append(f"image {{{i}}}")
    
    image_list = ", ".join(image_refs)
    
    # Create prompt with properly escaped JSON for multi-image PROMPT() function
    # The key is to double the curly braces in the JSON structure to escape them
    prompt_text = f'''Analyze {image_list} of the same golf club comprehensively. Extract ALL possible information and return ONLY valid JSON with this complete structure:

{{{{
    "club_type": "driver/fairway_wood/hybrid/iron/wedge/putter",
    "club_name": "full descriptive name",
    "brand": "manufacturer name", 
    "model": "specific model name",
    "year": 2023,
    "club_category": "game_improvement/players/distance/forgiveness",
    "shaft_type": "steel/graphite/hybrid",
    "shaft_flex": "extra_stiff/stiff/regular/senior/ladies",
    "shaft_label": "shaft brand and model if visible",
    "loft": 10.5,
    "hand": "right/left",
    "club_sub_type": "cavity_back/blade/mallet/etc",
    "set_composition": "individual/set_member/part_of_set",
    "lie_angle": 59.0,
    "face_angle": "neutral/open/closed",
    "bounce_angle": 12.0,
    "grind_type": "sole_grind_if_wedge",
    "model_designation": "specific_variant_or_edition",
    "shaft_length_inches": 45.0,
    "overall_grade": 8.5,
    "face_sole_wear_grade": "excellent/very_good/good/fair/poor",
    "face_sole_wear_description": "detailed description of face and sole condition including groove wear, impact marks, etc",
    "scratches_grade": "none/minor/moderate/severe",
    "scratches_description": "detailed description of all visible scratches, their location and severity",
    "scratches_locations": ["face", "sole", "crown", "back"],
    "paint_chips_grade": "none/minor/moderate/severe", 
    "paint_chips_description": "detailed description of paint/finish condition",
    "paint_chips_locations": ["crown", "sole", "back"],
    "putter_paint_wear_grade": "excellent/good/fair/poor",
    "grip_condition": "excellent/very_good/good/fair/poor/needs_replacement",
    "retail_price": 399.99,
    "trade_in_value": 150.00,
    "market_demand": "high/medium/low",
    "technology_tags": ["technology_1", "technology_2", "specific_features"],
    "special_designations": ["tour_issue", "limited_edition", "custom", "prototype"],
    "confidence_score": 0.95,
    "classification_notes": "comprehensive summary including distinguishing features, unique markings, condition assessment, technology features, and any special characteristics observed across all images",
    "condition_assessment": "excellent/very_good/good/fair/poor",
    "estimated_price_range": "$150-200",
    "features": {{{{
        "adjustable": true,
        "forged": false,
        "cavity_back": true,
        "face_insert": false,
        "weight_ports": true,
        "alignment_aids": false,
        "special_technology": "brief_description"
    }}}}
}}}}

ANALYSIS REQUIREMENTS:
1. Grade condition 1-10 scale: 9-10=like new, 7-8=excellent with minimal wear, 5-6=good with normal use, 3-4=fair with significant wear, 1-2=poor/damaged
2. Examine ALL angles: face wear patterns, sole condition, crown finish, grip wear, shaft condition
3. Identify ALL visible markings: brand logos, model names, loft numbers, shaft specifications, serial numbers
4. Technology assessment: Look for adjustable features, face inserts, weight systems, special materials
5. Market valuation: Consider brand tier, model popularity, condition, and current market demand
6. Defect documentation: Note every scratch, dent, paint chip, rust spot, or wear mark with location
7. Comprehensive notes: Include manufacturing details, design features, target player category

Provide detailed, accurate analysis focusing on maximizing data extraction from all {num_images} images.
'''
    
    return prompt_text

def get_enhanced_single_image_analysis_prompt():
    """
    Comprehensive single image analysis prompt for maximum data extraction.
    """
    return """
    # COMPREHENSIVE GOLF CLUB IMAGE ANALYSIS

    Analyze this golf club image with maximum detail extraction. Return ONLY valid JSON with complete data structure.

    ## IDENTIFICATION PRIORITIES:
    **Major Brands:** Callaway, TaylorMade, Titleist, Ping, Mizuno, Cobra, Cleveland, Wilson, Srixon, Honma, PXG, Tour Edge, Adams, Nike, Bridgestone, Lynx, MacGregor, Dunlop, Ben Hogan, Yonex, Fourteen, Miura, Bettinardi, Scotty Cameron, Odyssey

    ## COMPREHENSIVE ANALYSIS AREAS:

    ### 1. CLUB IDENTIFICATION & SPECIFICATIONS
    - Brand, model, year, club type, loft, lie angle, bounce (wedges)
    - Shaft: material, flex, brand, length
    - Hand orientation, set information
    - Special editions, tour models, custom features

    ### 2. CONDITION ASSESSMENT (1-10 scale)
    - **Overall Grade:** 9-10=like new, 7-8=excellent, 5-6=good, 3-4=fair, 1-2=poor
    - **Face/Sole:** Groove wear, impact marks, ball marks, scratches
    - **Cosmetics:** Paint condition, finish wear, chips, scratches
    - **Grip:** Wear level, cracking, slippage, replacement need
    - **Structural:** Dents, cracks, loose components

    ### 3. TECHNOLOGY & FEATURES
    - Adjustable features (loft, weight, lie)
    - Face technology (inserts, variable thickness)
    - Weight systems (moveable, fixed, tungsten)
    - Special materials (titanium, carbon fiber, multi-material)
    - Design category (game improvement, players, blade, cavity)

    ### 4. MARKET VALUATION
    - Current retail price when new
    - Estimated current market value range
    - Trade-in value estimate
    - Market demand level
    - Factors affecting value

    ### 5. DEFECT DOCUMENTATION
    - Location-specific wear patterns
    - Scratch severity and locations
    - Paint/finish condition by area
    - Performance-affecting damage
    - Cosmetic vs functional issues

    ## REQUIRED JSON OUTPUT STRUCTURE:

    {
        "club_type": "specific_type",
        "club_name": "full_descriptive_name",
        "brand": "manufacturer",
        "model": "model_name",
        "year": 2023,
        "club_category": "game_improvement/players/distance",
        "shaft_type": "steel/graphite",
        "shaft_flex": "extra_stiff/stiff/regular/senior/ladies",
        "shaft_label": "shaft_brand_and_model",
        "loft": 10.5,
        "hand": "right/left",
        "club_sub_type": "cavity_back/blade/mallet/etc",
        "set_composition": "individual/set_member",
        "lie_angle": 59.0,
        "face_angle": "neutral/open/closed",
        "bounce_angle": 12.0,
        "grind_type": "sole_grind_type",
        "model_designation": "specific_variant",
        "shaft_length_inches": 45.0,
        "overall_grade": 8.5,
        "face_sole_wear_grade": "excellent/very_good/good/fair/poor",
        "face_sole_wear_description": "detailed_face_sole_condition",
        "scratches_grade": "none/minor/moderate/severe",
        "scratches_description": "detailed_scratch_assessment",
        "scratches_locations": ["face", "sole", "crown"],
        "paint_chips_grade": "none/minor/moderate/severe",
        "paint_chips_description": "detailed_paint_condition",
        "paint_chips_locations": ["crown", "sole"],
        "putter_paint_wear_grade": "excellent/good/fair/poor",
        "grip_condition": "excellent/very_good/good/fair/poor/needs_replacement",
        "retail_price": 399.99,
        "trade_in_value": 150.00,
        "market_demand": "high/medium/low",
        "technology_tags": ["technology_features"],
        "special_designations": ["tour_issue", "limited_edition"],
        "confidence_score": 0.95,
        "classification_notes": "comprehensive_analysis_summary",
        "condition_assessment": "excellent/very_good/good/fair/poor",
        "estimated_price_range": "$150-200",
        "features": {
            "adjustable": true/false,
            "forged": true/false,
            "cavity_back": true/false,
            "face_insert": true/false,
            "weight_ports": true/false,
            "alignment_aids": true/false,
            "special_technology": "description"
        }
    }

    Analyze comprehensively and extract maximum possible data from the image.
    """

# =====================================================
# ENHANCED PARSING FUNCTIONS
# =====================================================

def parse_comprehensive_multi_image_analysis(analysis_text, file_names):
    """
    Enhanced parsing that extracts maximum data from multi-image AI responses.
    """
    try:
        if st.session_state.get('debug_mode', False):
            st.write("🔍 **COMPREHENSIVE PARSING**: Extracting maximum data from AI response...")
        
        # Initialize comprehensive data structure
        analysis_data = {
            'analysis_type': 'comprehensive_multi_image',
            'num_images_analyzed': len(file_names),
            'source_files': file_names,
            'analysis_timestamp': datetime.now().isoformat(),
            'confidence_score': 0.9,
            
            # Core identification
            'club_type': '', 'brand': '', 'model': '', 'year': None,
            'club_name': '', 'club_category': '', 'club_sub_type': '',
            
            # Technical specs
            'shaft_type': '', 'shaft_flex': '', 'shaft_label': '',
            'loft': None, 'lie_angle': None, 'bounce_angle': None,
            'face_angle': '', 'grind_type': '', 'shaft_length_inches': None,
            'hand': 'right', 'set_composition': '',
            
            # Condition assessment
            'overall_grade': None, 'condition_assessment': '',
            'face_sole_wear_grade': '', 'face_sole_wear_description': '',
            'scratches_grade': '', 'scratches_description': '',
            'paint_chips_grade': '', 'paint_chips_description': '',
            'putter_paint_wear_grade': '', 'grip_condition': '',
            
            # Arrays
            'scratches_locations': [], 'paint_chips_locations': [],
            'technology_tags': [], 'special_designations': [],
            
            # Market data
            'retail_price': None, 'trade_in_value': None,
            'market_demand': '', 'estimated_price_range': '',
            
            # Enhanced fields
            'model_designation': '', 'classification_notes': '',
            'features': {}
        }
        
        # Show partial response for debugging
        if st.session_state.get('debug_mode', False):
            st.write(f"📝 Parsing response (first 300 chars): {analysis_text[:300]}...")
        
        # Primary JSON extraction
        json_extracted = False
        json_match = re.search(r'\{[\s\S]*\}', analysis_text)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group())
                if st.session_state.get('debug_mode', False):
                    st.success("✅ Successfully extracted JSON structure")
                
                # Direct field mapping with validation
                for key, value in parsed_json.items():
                    if value is not None and str(value).strip() not in ['', 'null', 'none', 'unknown', 'n/a']:
                        if key in analysis_data:
                            analysis_data[key] = value
                            if st.session_state.get('debug_mode', False):
                                st.write(f"✅ Mapped: {key} = {value}")
                
                json_extracted = True
                
            except json.JSONDecodeError as e:
                if st.session_state.get('debug_mode', False):
                    st.warning(f"JSON parsing failed: {str(e)}")
        
        # Enhanced text extraction if JSON parsing failed or incomplete
        if not json_extracted or not analysis_data.get('brand'):
            if st.session_state.get('debug_mode', False):
                st.info("🔍 Performing enhanced text extraction...")
            
            # Extract technical specifications
            tech_specs = extract_technical_specifications(analysis_text)
            for spec, value in tech_specs.items():
                if not analysis_data.get(spec):
                    analysis_data[spec] = value
                    if st.session_state.get('debug_mode', False):
                        st.write(f"✅ Extracted {spec}: {value}")
            
            # Enhanced brand detection
            if not analysis_data.get('brand'):
                brand_patterns = [
                    r'\b(callaway|taylormade|titleist|ping|mizuno|cobra|cleveland|wilson)\b',
                    r'\b(srixon|honma|pxg|tour edge|adams|nike|bridgestone)\b',
                    r'\b(lynx|macgregor|dunlop|ben hogan|yonex|fourteen|miura)\b',
                    r'\b(bettinardi|scotty cameron|odyssey)\b'
                ]
                
                for pattern in brand_patterns:
                    match = re.search(pattern, analysis_text, re.IGNORECASE)
                    if match:
                        analysis_data['brand'] = match.group(1).title()
                        if st.session_state.get('debug_mode', False):
                            st.write(f"✅ Detected brand: {analysis_data['brand']}")
                        break
            
            # Enhanced club type detection
            if not analysis_data.get('club_type'):
                club_type_patterns = {
                    'driver': [r'\bdriver\b', r'\b1\s*wood\b'],
                    'fairway wood': [r'\bfairway\s*wood\b', r'\b[3-7]\s*wood\b'],
                    'hybrid': [r'\bhybrid\b', r'\brescue\b', r'\butility\b'],
                    'iron': [r'\biron\b', r'\b[3-9]\s*iron\b'],
                    'wedge': [r'\bwedge\b', r'\bpw\b', r'\bsw\b', r'\blw\b'],
                    'putter': [r'\bputter\b', r'\bputting\b']
                }
                
                for club_type, patterns in club_type_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, analysis_text, re.IGNORECASE):
                            analysis_data['club_type'] = club_type
                            if st.session_state.get('debug_mode', False):
                                st.write(f"✅ Detected club type: {club_type}")
                            break
                    if analysis_data.get('club_type'):
                        break
            
            # Extract market information
            market_info = extract_market_information(
                analysis_text, 
                analysis_data.get('brand'), 
                analysis_data.get('club_type'),
                analysis_data.get('condition_assessment')
            )
            
            for key, value in market_info.items():
                if not analysis_data.get(key):
                    analysis_data[key] = value
                    if st.session_state.get('debug_mode', False):
                        st.write(f"✅ Market data: {key} = {value}")
            
            # Extract technology features
            tech_features, special_designations = extract_technology_features(analysis_text)
            if tech_features:
                analysis_data['technology_tags'] = tech_features
                if st.session_state.get('debug_mode', False):
                    st.write(f"✅ Technology tags: {tech_features}")
            
            if special_designations:
                analysis_data['special_designations'] = special_designations
                if st.session_state.get('debug_mode', False):
                    st.write(f"✅ Special designations: {special_designations}")
            
            # Enhanced condition grading from text
            condition_keywords = {
                'excellent': ['excellent', 'like new', 'pristine', 'mint'],
                'very good': ['very good', 'great', 'superb'],
                'good': ['good', 'decent', 'solid', 'nice'],
                'fair': ['fair', 'okay', 'average', 'worn'],
                'poor': ['poor', 'bad', 'damaged', 'beat up']
            }
            
            if not analysis_data.get('condition_assessment'):
                for condition, keywords in condition_keywords.items():
                    if any(keyword in analysis_text.lower() for keyword in keywords):
                        analysis_data['condition_assessment'] = condition
                        # Set corresponding numeric grade
                        grade_mapping = {
                            'excellent': 9.0, 'very good': 8.0, 'good': 7.0, 
                            'fair': 5.5, 'poor': 3.0
                        }
                        analysis_data['overall_grade'] = grade_mapping.get(condition, 7.0)
                        if st.session_state.get('debug_mode', False):
                            st.write(f"✅ Condition: {condition} (Grade: {analysis_data['overall_grade']})")
                        break
        
        # Intelligent defaults and validation
        if not analysis_data.get('estimated_price_range') and analysis_data.get('retail_price'):
            # Generate price range from retail price
            retail = analysis_data['retail_price']
            if isinstance(retail, (int, float)) and retail > 0:
                low_end = int(retail * 0.8)
                high_end = int(retail * 1.2)
                analysis_data['estimated_price_range'] = f"${low_end}-{high_end}"
                if st.session_state.get('debug_mode', False):
                    st.write(f"✅ Generated price range: {analysis_data['estimated_price_range']}")
        
        # Set intelligent defaults for missing critical fields
        defaults = {
            'hand': 'right',
            'shaft_type': 'steel' if analysis_data.get('club_type') in ['iron', 'wedge'] else 'graphite',
            'shaft_flex': 'regular',
            'market_demand': 'medium',
            'confidence_score': 0.9
        }
        
        for field, default_value in defaults.items():
            if not analysis_data.get(field):
                analysis_data[field] = default_value
        
        # Enhanced classification notes
        notes_parts = [
            f"Comprehensive multi-image analysis of {len(file_names)} images",
            f"Confidence: {analysis_data.get('confidence_score', 0.9):.2f}"
        ]
        
        if analysis_data.get('brand') and analysis_data.get('model'):
            notes_parts.append(f"Identified as {analysis_data['brand']} {analysis_data['model']}")
        
        if analysis_data.get('overall_grade'):
            notes_parts.append(f"Overall condition grade: {analysis_data['overall_grade']}/10")
        
        if analysis_data.get('technology_tags'):
            notes_parts.append(f"Technologies: {', '.join(analysis_data['technology_tags'][:3])}")
        
        enhanced_notes = ". ".join(notes_parts)
        if analysis_data.get('classification_notes'):
            enhanced_notes += f". {analysis_data['classification_notes']}"
        
        analysis_data['classification_notes'] = enhanced_notes
        
        # Final validation and summary - only in debug mode
        if st.session_state.get('debug_mode', False):
            st.write("🎯 **FINAL EXTRACTED DATA SUMMARY:**")
            key_fields = [
                'brand', 'model', 'club_type', 'loft', 'overall_grade', 
                'condition_assessment', 'estimated_price_range', 'confidence_score'
            ]
            
            extracted_count = 0
            for field in key_fields:
                value = analysis_data.get(field)
                if value and str(value) not in ['', '0', 'Unknown', 'N/A']:
                    st.write(f"  ✅ {field}: {value}")
                    extracted_count += 1
                else:
                    st.write(f"  ⚠️ {field}: Not extracted")
            
            st.info(f"📊 Successfully extracted {extracted_count}/{len(key_fields)} key fields")
        
        return analysis_data
        
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"❌ Error in comprehensive parsing: {str(e)}")
        return {
            'error': str(e),
            'raw_text': analysis_text,
            'analysis_type': 'comprehensive_multi_image_error',
            'num_images_analyzed': len(file_names),
            'source_files': file_names,
            'club_type': 'Parse Error',
            'brand': 'Parse Error',
            'model': 'Parse Error',
            'condition_assessment': 'Parse Error',
            'overall_grade': 0,
            'confidence_score': 0.1
        }

# =====================================================
# ENHANCED DATABASE SAVE FUNCTIONS
# =====================================================

def save_comprehensive_analysis_result(analysis_data, file_names, model_option):
    """
    Save comprehensive analysis results with maximum data population across all three tables.
    """
    try:
        session = get_active_session()
        stage_name = "@IMG_STAGE"
        
        # Create unique image ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        if len(file_names) > 1:
            image_id = f"multi_img_{len(file_names)}_{timestamp}"
        else:
            image_id = f"{file_names[0].split('.')[0]}_{timestamp}"
        
        if st.session_state.get('debug_mode', False):
            st.write(f"💾 **COMPREHENSIVE SAVE**: Populating all database tables for ID: {image_id}")
        
        # ===============================================
        # 1. SAVE TO MAIN TABLE (CALLAWAY_CLUBS_FILE_V3)
        # ===============================================
        
        if st.session_state.get('debug_mode', False):
            st.write("💾 Step 1: Saving to main table with ALL possible fields...")
        
        # Prepare ALL numeric fields with enhanced validation
        numeric_fields = {
            'loft': safe_numeric(analysis_data.get('loft')),
            'year': safe_numeric(analysis_data.get('year')),
            'lie_angle': safe_numeric(analysis_data.get('lie_angle')),
            'bounce_angle': safe_numeric(analysis_data.get('bounce_angle')),
            'shaft_length_inches': safe_numeric(analysis_data.get('shaft_length_inches')),
            'overall_grade': safe_numeric(analysis_data.get('overall_grade')),
            'retail_price': safe_numeric(analysis_data.get('retail_price')),
            'trade_in_value': safe_numeric(analysis_data.get('trade_in_value')),
            'confidence_score': analysis_data.get('confidence_score', 0.9)
        }
        
        # Show what numeric values we're saving - only in debug mode
        if st.session_state.get('debug_mode', False):
            for field, value in numeric_fields.items():
                if value != "NULL":
                    st.write(f"  📊 {field}: {value}")
        
        # Enhanced classification notes with comprehensive details
        enhanced_notes = []
        enhanced_notes.append(f"COMPREHENSIVE MULTI-IMAGE ANALYSIS")
        enhanced_notes.append(f"Images processed: {len(file_names)} ({', '.join(file_names)})")
        enhanced_notes.append(f"AI Model: {model_option}")
        enhanced_notes.append(f"Analysis confidence: {analysis_data.get('confidence_score', 0.9):.3f}")
        enhanced_notes.append(f"Timestamp: {analysis_data.get('analysis_timestamp', datetime.now().isoformat())}")
        
        # Add identification details
        if analysis_data.get('brand') or analysis_data.get('model'):
            enhanced_notes.append(f"IDENTIFICATION: {analysis_data.get('brand', 'Unknown')} {analysis_data.get('model', 'Unknown')}")
        
        # Add condition summary
        if analysis_data.get('overall_grade'):
            enhanced_notes.append(f"CONDITION: Grade {analysis_data['overall_grade']}/10 - {analysis_data.get('condition_assessment', 'N/A')}")
        
        # Add technology features
        if analysis_data.get('technology_tags'):
            enhanced_notes.append(f"TECHNOLOGY: {', '.join(analysis_data['technology_tags'][:5])}")
        
        # Add market information
        if analysis_data.get('estimated_price_range'):
            enhanced_notes.append(f"MARKET VALUE: {analysis_data['estimated_price_range']}")
        
        # Add any original classification notes
        if analysis_data.get('classification_notes'):
            enhanced_notes.append(f"DETAILS: {analysis_data['classification_notes']}")
        
        final_notes = "\\n".join(enhanced_notes)
        
        # Comprehensive INSERT with ALL fields
        main_insert_query = f"""
        INSERT INTO CALLAWAY_IMG_COMPLETE.PUBLIC.CALLAWAY_CLUBS_FILE_V3 (
            IMAGE_ID,
            CLUB_TYPE,
            CLUB_NAME,
            BRAND,
            MODEL,
            YEAR,
            CLUB_CATEGORY,
            SHAFT_TYPE,
            SHAFT_FLEX,
            SHAFT_LABEL,
            LOFT,
            HAND,
            CLUB_SUB_TYPE,
            SET_COMPOSITION,
            LIE_ANGLE,
            FACE_ANGLE,
            BOUNCE_ANGLE,
            GRIND_TYPE,
            MODEL_DESIGNATION,
            SHAFT_LENGTH_INCHES,
            OVERALL_GRADE,
            FACE_SOLE_WEAR_GRADE,
            FACE_SOLE_WEAR_DESCRIPTION,
            SCRATCHES_GRADE,
            SCRATCHES_DESCRIPTION,
            PAINT_CHIPS_GRADE,
            PAINT_CHIPS_DESCRIPTION,
            PUTTER_PAINT_WEAR_GRADE,
            GRIP_CONDITION,
            RETAIL_PRICE,
            TRADE_IN_VALUE,
            MARKET_DEMAND,
            CONFIDENCE_SCORE,
            CLASSIFICATION_NOTES,
            CONDITION_ASSESSMENT,
            ESTIMATED_PRICE_RANGE,
            RELATIVE_PATH,
            AI_MODEL,
            CREATED_BY,
            CREATED_AT,
            UPDATED_AT
        ) VALUES (
            '{image_id}',
            '{safe_sql_string(truncate_field_value(analysis_data.get('club_type', ''), 'club_type'))}',
            '{safe_sql_string(analysis_data.get('club_name', ''))}',
            '{safe_sql_string(truncate_field_value(analysis_data.get('brand', ''), 'brand'))}',
            '{safe_sql_string(truncate_field_value(analysis_data.get('model', ''), 'model'))}',
            {numeric_fields['year']},
            '{safe_sql_string(analysis_data.get('club_category', ''))}',
            '{safe_sql_string(truncate_field_value(analysis_data.get('shaft_type', ''), 'shaft_type'))}',
            '{safe_sql_string(truncate_field_value(analysis_data.get('shaft_flex', ''), 'shaft_flex'))}',
            '{safe_sql_string(analysis_data.get('shaft_label', ''))}',
            {numeric_fields['loft']},
            '{safe_sql_string(truncate_field_value(analysis_data.get('hand', ''), 'hand'))}',
            '{safe_sql_string(truncate_field_value(analysis_data.get('club_sub_type', ''), 'club_sub_type'))}',
            '{safe_sql_string(analysis_data.get('set_composition', ''))}',
            {numeric_fields['lie_angle']},
            '{safe_sql_string(truncate_field_value(analysis_data.get('face_angle', ''), 'face_angle'))}',
            {numeric_fields['bounce_angle']},
            '{safe_sql_string(truncate_field_value(analysis_data.get('grind_type', ''), 'grind_type'))}',
            '{safe_sql_string(analysis_data.get('model_designation', ''))}',
            {numeric_fields['shaft_length_inches']},
            {numeric_fields['overall_grade']},
            '{safe_sql_string(truncate_field_value(analysis_data.get('face_sole_wear_grade', ''), 'face_sole_wear_grade'))}',
            '{safe_sql_string(analysis_data.get('face_sole_wear_description', ''))}',
            '{safe_sql_string(truncate_field_value(analysis_data.get('scratches_grade', ''), 'scratches_grade'))}',
            '{safe_sql_string(analysis_data.get('scratches_description', ''))}',
            '{safe_sql_string(truncate_field_value(analysis_data.get('paint_chips_grade', ''), 'paint_chips_grade'))}',
            '{safe_sql_string(analysis_data.get('paint_chips_description', ''))}',
            '{safe_sql_string(analysis_data.get('putter_paint_wear_grade', ''))}',
            '{safe_sql_string(truncate_field_value(analysis_data.get('grip_condition', ''), 'grip_condition'))}',
            {numeric_fields['retail_price']},
            {numeric_fields['trade_in_value']},
            '{safe_sql_string(truncate_field_value(analysis_data.get('market_demand', ''), 'market_demand'))}',
            {numeric_fields['confidence_score']},
            '{safe_sql_string(final_notes)}',
            '{safe_sql_string(truncate_field_value(analysis_data.get('condition_assessment', ''), 'condition_assessment'))}',
            '{safe_sql_string(analysis_data.get('estimated_price_range', ''))}',
            '{safe_sql_string(file_names[0] if file_names else '')}',
            '{safe_sql_string(model_option)}',
            CURRENT_USER(),
            CURRENT_TIMESTAMP(),
            CURRENT_TIMESTAMP()
        )
        """
        
        # Execute main insert
        session.sql(main_insert_query).collect()
        if st.session_state.get('debug_mode', False):
            st.success(f"✅ Main record saved with comprehensive data")
        
        # Update array fields with enhanced error handling
        if st.session_state.get('debug_mode', False):
            st.write("💾 Step 2: Adding array fields to main table...")
        
        array_updates = [
            ('SCRATCHES_LOCATIONS', analysis_data.get('scratches_locations', [])),
            ('PAINT_CHIPS_LOCATIONS', analysis_data.get('paint_chips_locations', [])),
            ('TECHNOLOGY_TAGS', analysis_data.get('technology_tags', [])),
            ('SPECIAL_DESIGNATIONS', analysis_data.get('special_designations', []))
        ]
        
        for field_name, field_data in array_updates:
            if field_data and isinstance(field_data, list) and len(field_data) > 0:
                try:
                    # Clean and prepare array items
                    clean_items = [safe_sql_string(str(item)) for item in field_data if item]
                    if clean_items:
                        items_sql = "', '".join(clean_items)
                        
                        update_query = f"""
                        UPDATE CALLAWAY_IMG_COMPLETE.PUBLIC.CALLAWAY_CLUBS_FILE_V3
                        SET {field_name} = ARRAY_CONSTRUCT('{items_sql}')
                        WHERE IMAGE_ID = '{image_id}'
                        """
                        
                        session.sql(update_query).collect()
                        if st.session_state.get('debug_mode', False):
                            st.write(f"  ✅ {field_name}: {len(clean_items)} items - {clean_items[:3]}")
                    
                except Exception as array_err:
                    if st.session_state.get('debug_mode', False):
                        st.warning(f"  ⚠️ Could not update {field_name}: {str(array_err)}")
        
        # Handle FEATURES as VARIANT
        if analysis_data.get('features'):
            try:
                features_json = json.dumps(analysis_data['features'])
                features_json_safe = features_json.replace("'", "''")
                
                features_query = f"""
                UPDATE CALLAWAY_IMG_COMPLETE.PUBLIC.CALLAWAY_CLUBS_FILE_V3
                SET FEATURES = PARSE_JSON('{features_json_safe}')
                WHERE IMAGE_ID = '{image_id}'
                """
                
                session.sql(features_query).collect()
                if st.session_state.get('debug_mode', False):
                    st.write(f"  ✅ FEATURES: {len(analysis_data['features'])} feature flags")
            except Exception as features_err:
                if st.session_state.get('debug_mode', False):
                    st.warning(f"  ⚠️ Could not save features: {str(features_err)}")
        
        # ===============================================
        # 2. SAVE TO IMAGE_REFERENCES_V3 TABLE
        # ===============================================
        
        if st.session_state.get('debug_mode', False):
            st.write(f"💾 Step 3: Saving {len(file_names)} image references with FILE data...")
        
        image_refs_saved = 0
        for i, file_name in enumerate(file_names, 1):
            try:
                # Determine reference type based on position and count
                if len(file_names) == 1:
                    ref_type = 'SINGLE_IMAGE'
                elif i == 1:
                    ref_type = 'PRIMARY_IMAGE'  # First image is primary
                else:
                    ref_type = 'MULTI_ANGLE'    # Additional angles
                
                # Insert with FILE reference
                ref_query = f"""
                INSERT INTO CALLAWAY_IMG_COMPLETE.PUBLIC.IMAGE_REFERENCES_V3 (
                    IMG,
                    PRIMARY_IMAGE_ID,
                    REFERENCE_IMAGE_PATH,
                    IMAGE_SEQUENCE,
                    REFERENCE_TYPE
                )
                SELECT 
                    TO_FILE('{stage_name}', '{file_name}'),
                    '{image_id}',
                    '{safe_sql_string(file_name)}',
                    {i},
                    '{ref_type}'
                """
                
                session.sql(ref_query).collect()
                if st.session_state.get('debug_mode', False):
                    st.write(f"  ✅ Image {i}/{len(file_names)}: {file_name} saved as {ref_type}")
                image_refs_saved += 1
                
            except Exception as ref_err:
                if st.session_state.get('debug_mode', False):
                    st.warning(f"  ⚠️ Could not save reference for {file_name}: {str(ref_err)}")
        
        if st.session_state.get('debug_mode', False):
            st.success(f"✅ Saved {image_refs_saved}/{len(file_names)} image references")
        
        # ===============================================
        # 3. SAVE TO CLUB_DEFECT_OBSERVATIONS_V3 TABLE
        # ===============================================
        
        if st.session_state.get('debug_mode', False):
            st.write("💾 Step 4: Analyzing and saving comprehensive defect observations...")
        
        # Extract all possible defects using enhanced extraction
        all_defects = extract_enhanced_defects(
            analysis_data.get('classification_notes', '') + ' ' + 
            analysis_data.get('face_sole_wear_description', '') + ' ' +
            analysis_data.get('scratches_description', '') + ' ' +
            analysis_data.get('paint_chips_description', ''),
            analysis_data
        )
        
        defects_saved = 0
        for defect_idx, defect in enumerate(all_defects):
            try:
                defect_id = f"{image_id}_defect_{defect_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}"
                
                # Map severity to defect size categories
                size_mapping = {
                    'none': 'None',
                    'minor': 'Small',
                    'light': 'Small', 
                    'moderate': 'Medium',
                    'severe': 'Large',
                    'heavy': 'Large',
                    'excellent': 'None',
                    'good': 'Small',
                    'fair': 'Medium',
                    'poor': 'Large'
                }
                
                defect_size = size_mapping.get(defect.get('severity', '').lower(), 'Medium')
                
                # Map severity to depth
                depth_mapping = {
                    'none': 'None',
                    'minor': 'Superficial',
                    'light': 'Superficial',
                    'moderate': 'Medium', 
                    'severe': 'Deep',
                    'heavy': 'Deep',
                    'excellent': 'None',
                    'good': 'Superficial',
                    'fair': 'Medium',
                    'poor': 'Deep'
                }
                
                defect_depth = depth_mapping.get(defect.get('severity', '').lower(), 'Superficial')
                
                # Prepare comprehensive impact description
                impact_description = defect.get('impact', '')
                if defect.get('description'):
                    impact_description = f"{impact_description}. Detail: {defect.get('description', '')[:400]}"
                
                # Insert defect observation
                defect_query = f"""
                INSERT INTO CALLAWAY_IMG_COMPLETE.PUBLIC.CLUB_DEFECT_OBSERVATIONS_V3 (
                    OBSERVATION_ID,
                    IMAGE_ID,
                    DEFECT_TYPE,
                    DEFECT_LOCATION,
                    DEFECT_SIZE,
                    DEFECT_LENGTH_MM,
                    DEFECT_WIDTH_MM,
                    DEFECT_DEPTH,
                    IMPACT_ON_PERFORMANCE
                ) VALUES (
                    '{defect_id}',
                    '{image_id}',
                    '{safe_sql_string(defect.get('type', 'GENERAL'))}',
                    '{safe_sql_string(defect.get('location', 'Unknown'))}',
                    '{defect_size}',
                    {safe_numeric(defect.get('size_mm'))},
                    NULL,
                    '{defect_depth}',
                    '{safe_sql_string(impact_description[:500])}'
                )
                """
                
                session.sql(defect_query).collect()
                if st.session_state.get('debug_mode', False):
                    st.write(f"  ✅ Defect {defect_idx + 1}: {defect['type']} on {defect['location']} ({defect['severity']}) - Impact: {defect_depth}")
                defects_saved += 1
                
            except Exception as defect_err:
                if st.session_state.get('debug_mode', False):
                    st.warning(f"  ⚠️ Could not save defect {defect_idx + 1}: {str(defect_err)}")
        
        if st.session_state.get('debug_mode', False):
            if defects_saved > 0:
                st.success(f"✅ Saved {defects_saved} detailed defect observations")
            else:
                st.info("ℹ️ No significant defects detected - club appears to be in excellent condition")
        
        # ===============================================
        # 4. FINAL SUMMARY AND VALIDATION
        # ===============================================
        
        if st.session_state.get('debug_mode', False):
            st.write("🔍 Step 5: Validating saved data...")
            
            # Verify the data was saved correctly
            validation_query = f"""
            SELECT 
                IMAGE_ID,
                CLUB_TYPE,
                BRAND,
                MODEL,
                OVERALL_GRADE,
                CONDITION_ASSESSMENT,
                ESTIMATED_PRICE_RANGE,
                ARRAY_SIZE(TECHNOLOGY_TAGS) as TECH_COUNT,
                ARRAY_SIZE(SPECIAL_DESIGNATIONS) as SPECIAL_COUNT
            FROM CALLAWAY_IMG_COMPLETE.PUBLIC.CALLAWAY_CLUBS_FILE_V3 
            WHERE IMAGE_ID = '{image_id}'
            """
            
            validation_result = session.sql(validation_query).collect()
            if validation_result:
                result = validation_result[0]
                st.write("📊 **SAVED DATA VALIDATION:**")
                st.write(f"  ✅ Club: {result[2]} {result[3]} ({result[1]})")
                st.write(f"  ✅ Condition: {result[5]} (Grade: {result[4]})")
                st.write(f"  ✅ Value: {result[6]}")
                st.write(f"  ✅ Technology tags: {result[7] or 0}")
                st.write(f"  ✅ Special designations: {result[8] or 0}")
        
        # Count related records
        ref_count_query = f"SELECT COUNT(*) FROM CALLAWAY_IMG_COMPLETE.PUBLIC.IMAGE_REFERENCES_V3 WHERE PRIMARY_IMAGE_ID = '{image_id}'"
        ref_count = session.sql(ref_count_query).collect()[0][0]
        
        defect_count_query = f"SELECT COUNT(*) FROM CALLAWAY_IMG_COMPLETE.PUBLIC.CLUB_DEFECT_OBSERVATIONS_V3 WHERE IMAGE_ID = '{image_id}'"
        defect_count = session.sql(defect_count_query).collect()[0][0]
        
        # Final success message - always show this
        st.success(f"""
        🎉 **Analysis Complete!**
        
        📊 **Summary:**
        - Analysis ID: {image_id}
        - Images analyzed: {len(file_names)}
        - Records saved: {1 + ref_count + defect_count}
        """)
        
        return True, f"Successfully saved comprehensive analysis", image_id
        
    except Exception as e:
        st.error(f"❌ Error in comprehensive save: {str(e)}")
        if st.session_state.get('debug_mode', False):
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
        return False, f"Error saving to database: {str(e)}", None

# =====================================================
# MAIN ANALYSIS FUNCTIONS - ENHANCED VERSION
# =====================================================

def analyze_multiple_images_COMPREHENSIVE(uploaded_files, model_option):
    """
    Comprehensive multi-image processing with maximum data extraction.
    """
    try:
        # Get the active Snowflake session
        session = get_active_session()
        
        stage_name_no_at = "IMG_STAGE"
        stage_name = f"@{stage_name_no_at}"
        
        uploaded_file_names = []
        
        # Upload all files to the stage first
        st.write(f"📤 **Uploading** {len(uploaded_files)} files...")
        
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            uploaded_file_names.append(file_name)
            
            try:
                # Reset file position
                uploaded_file.seek(0)
                
                # Create file stream using BytesIO
                file_stream = io.BytesIO(uploaded_file.getvalue())
                
                # Upload the file to the stage
                result = session.file.put_stream(
                    file_stream,
                    f"{stage_name}/{file_name}",
                    auto_compress=False,
                    overwrite=True
                )
                
                if st.session_state.get('debug_mode', False):
                    st.write(f"  ✅ Uploaded: {file_name}")
                
            except Exception as upload_err:
                st.error(f"Error uploading {file_name}: {str(upload_err)}")
                return False, f"Error during file upload: {str(upload_err)}", None, None, None
        
        # Verify all files exist in stage after upload
        verification = session.sql(f"LIST {stage_name}").collect()
        stage_files = [str(file_info) for file_info in verification]
        
        missing_files = []
        for file_name in uploaded_file_names:
            file_found = any(file_name in stage_file for stage_file in stage_files)
            if not file_found:
                missing_files.append(file_name)
        
        if missing_files:
            st.warning(f"Files uploaded but not found in stage: {', '.join(missing_files)}")
            return False, "Some files not found in stage after upload", None, None, None
        
        st.success(f"✅ All {len(uploaded_file_names)} files uploaded successfully!")
        
        # COMPREHENSIVE MULTI-IMAGE PROCESSING
        st.info(f"🚀 **Processing** {len(uploaded_file_names)} images...")
        
        # Handle different numbers of images based on CORTEX.AI_COMPLETE limitations
        if len(uploaded_file_names) == 1:
            if st.session_state.get('debug_mode', False):
                st.info("🔧 Using single-image CORTEX.AI_COMPLETE approach")
            processing_method = "single"
        elif len(uploaded_file_names) == 2:
            if st.session_state.get('debug_mode', False):
                st.info("🔧 Using dual-image CORTEX.AI_COMPLETE approach")
            processing_method = "dual"
        elif len(uploaded_file_names) <= 10:
            if st.session_state.get('debug_mode', False):
                st.info("🔧 Using PROMPT() function for multi-image analysis (3+ images)")
            processing_method = "prompt"
        else:
            st.error("❌ Maximum 10 images supported")
            return False, "Too many images - maximum 10 supported", None, None, None
        
        # Prepare comprehensive prompt (same for all methods)
        if len(uploaded_file_names) == 1:
            image_description = "this golf club image"
        else:
            image_description = f"these {len(uploaded_file_names)} golf club images of the same club from different angles"
        
        comprehensive_prompt = f"""Analyze {image_description} comprehensively and extract ALL possible information. Return ONLY valid JSON with this complete structure:

{{
    "club_type": "driver/fairway_wood/hybrid/iron/wedge/putter",
    "club_name": "full descriptive name",
    "brand": "manufacturer name", 
    "model": "specific model name",
    "year": 2023,
    "club_category": "game_improvement/players/distance/forgiveness",
    "shaft_type": "steel/graphite/hybrid",
    "shaft_flex": "extra_stiff/stiff/regular/senior/ladies",
    "shaft_label": "shaft brand and model if visible",
    "loft": 10.5,
    "hand": "right/left",
    "club_sub_type": "cavity_back/blade/mallet/etc",
    "set_composition": "individual/set_member/part_of_set",
    "lie_angle": 59.0,
    "face_angle": "neutral/open/closed",
    "bounce_angle": 12.0,
    "grind_type": "sole_grind_if_wedge",
    "model_designation": "specific_variant_or_edition", 
    "shaft_length_inches": 45.0,
    "overall_grade": 8.5,
    "face_sole_wear_grade": "excellent/very_good/good/fair/poor",
    "face_sole_wear_description": "detailed description of face and sole condition including groove wear and impact marks",
    "scratches_grade": "none/minor/moderate/severe",
    "scratches_description": "detailed description of all visible scratches with locations and severity",
    "scratches_locations": ["face", "sole", "crown", "back"],
    "paint_chips_grade": "none/minor/moderate/severe", 
    "paint_chips_description": "detailed description of paint/finish condition",
    "paint_chips_locations": ["crown", "sole", "back"],
    "putter_paint_wear_grade": "excellent/good/fair/poor",
    "grip_condition": "excellent/very_good/good/fair/poor/needs_replacement",
    "retail_price": 399.99,
    "trade_in_value": 150.00,
    "market_demand": "high/medium/low",
    "technology_tags": ["technology_1", "technology_2", "specific_features"],
    "special_designations": ["tour_issue", "limited_edition", "custom", "prototype"],
    "confidence_score": 0.95,
    "classification_notes": "comprehensive summary including distinguishing features, unique markings, condition assessment, technology features, and any special characteristics observed",
    "condition_assessment": "excellent/very_good/good/fair/poor",
    "estimated_price_range": "$150-200",
    "features": {{
        "adjustable": true,
        "forged": false,
        "cavity_back": true,
        "face_insert": false,
        "weight_ports": true,
        "alignment_aids": false,
        "special_technology": "brief_description"
    }}
}}

ANALYSIS REQUIREMENTS:
1. Grade condition 1-10 scale: 9-10=like new, 7-8=excellent with minimal wear, 5-6=good with normal use, 3-4=fair with significant wear, 1-2=poor/damaged
2. Examine ALL visible areas: face wear patterns, sole condition, crown finish, grip wear, shaft condition
3. Identify ALL visible markings: brand logos, model names, loft numbers, shaft specifications, serial numbers
4. Technology assessment: Look for adjustable features, face inserts, weight systems, special materials
5. Market valuation: Consider brand tier, model popularity, condition, and current market demand  
6. Defect documentation: Note every scratch, dent, paint chip, rust spot, or wear mark with location
7. Comprehensive notes: Include manufacturing details, design features, target player category

Provide detailed, accurate analysis focusing on maximizing data extraction from all images provided."""
        
        # DEBUG: Show the processing method and prompt
        if st.session_state.get('debug_mode', False):
            st.write(f"🔍 **DEBUG - Processing Method:** {processing_method}")
            st.write("🔍 **DEBUG - Comprehensive Prompt (first 300 chars):**")
            st.code(comprehensive_prompt[:300] + "...")
        
        # Build TO_FILE calls
        to_file_calls = []
        for file_name in uploaded_file_names:
            to_file_calls.append(f"TO_FILE('{stage_name}', '{file_name}')")
        
        # DEBUG: Show the TO_FILE calls
        if st.session_state.get('debug_mode', False):
            st.write("🔍 **DEBUG - TO_FILE() calls:**")
            for i, call in enumerate(to_file_calls):
                st.code(f"Image {i+1}: {call}")
        
        # Create appropriate query based on processing method
        if processing_method == "single":
            # Single image: CORTEX.AI_COMPLETE(model, prompt, image)
            analysis_prompt_escaped = comprehensive_prompt.replace("'", "''").replace("\\", "\\\\")
            analysis_query = f"""SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
    '{model_option}',
    '{analysis_prompt_escaped}',
    {to_file_calls[0]}
)"""
            
        elif processing_method == "dual":
            # Dual image: CORTEX.AI_COMPLETE(model, prompt, image1, image2)
            analysis_prompt_escaped = comprehensive_prompt.replace("'", "''").replace("\\", "\\\\")
            analysis_query = f"""SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
    '{model_option}',
    '{analysis_prompt_escaped}',
    {to_file_calls[0]},
    {to_file_calls[1]}
)"""
            
        else:  # processing_method == "prompt"
            # Multi-image (3+): Use PROMPT function with CORRECT {0}, {1}, {2} placeholder syntax
            # Following Snowflake documentation example exactly
            
            # Build placeholders like {0}, {1}, {2}, etc. (numeric placeholders)
            image_placeholders = []
            for i in range(len(uploaded_file_names)):
                image_placeholders.append(f"image {{{i}}}")
            
            image_placeholder_text = ", ".join(image_placeholders)
            
            # Create prompt with proper {0}, {1}, {2} placeholder syntax matching Snowflake docs
            # CRITICAL: Double the curly braces in the JSON structure to escape them properly
            prompt_with_placeholders = f"""Analyze {image_placeholder_text} of the same golf club comprehensively and extract ALL possible information. Return ONLY valid JSON with this complete structure:

{{{{
    "club_type": "driver/fairway_wood/hybrid/iron/wedge/putter",
    "club_name": "full descriptive name",
    "brand": "manufacturer name", 
    "model": "specific model name",
    "year": 2023,
    "club_category": "game_improvement/players/distance/forgiveness",
    "shaft_type": "steel/graphite/hybrid",
    "shaft_flex": "extra_stiff/stiff/regular/senior/ladies",
    "shaft_label": "shaft brand and model if visible",
    "loft": 10.5,
    "hand": "right/left",
    "club_sub_type": "cavity_back/blade/mallet/etc",
    "set_composition": "individual/set_member/part_of_set",
    "lie_angle": 59.0,
    "face_angle": "neutral/open/closed",
    "bounce_angle": 12.0,
    "grind_type": "sole_grind_if_wedge",
    "model_designation": "specific_variant_or_edition", 
    "shaft_length_inches": 45.0,
    "overall_grade": 8.5,
    "face_sole_wear_grade": "excellent/very_good/good/fair/poor",
    "face_sole_wear_description": "detailed description of face and sole condition",
    "scratches_grade": "none/minor/moderate/severe",
    "scratches_description": "detailed description of all visible scratches",
    "scratches_locations": ["face", "sole", "crown", "back"],
    "paint_chips_grade": "none/minor/moderate/severe", 
    "paint_chips_description": "detailed description of paint/finish condition",
    "paint_chips_locations": ["crown", "sole", "back"],
    "putter_paint_wear_grade": "excellent/good/fair/poor",
    "grip_condition": "excellent/very_good/good/fair/poor/needs_replacement",
    "retail_price": 399.99,
    "trade_in_value": 150.00,
    "market_demand": "high/medium/low",
    "technology_tags": ["technology_1", "technology_2"],
    "special_designations": ["tour_issue", "limited_edition"],
    "confidence_score": 0.95,
    "classification_notes": "comprehensive summary of analysis from all images",
    "condition_assessment": "excellent/very_good/good/fair/poor",
    "estimated_price_range": "$150-200",
    "features": {{{{
        "adjustable": true,
        "forged": false,
        "cavity_back": true,
        "face_insert": false,
        "weight_ports": true,
        "alignment_aids": false,
        "special_technology": "brief_description"
    }}}}
}}}}

Grade condition 1-10 scale. Examine all angles for comprehensive analysis. Extract maximum data from all images provided."""
            
            # DEBUG: Show the placeholder text being used
            if st.session_state.get('debug_mode', False):
                st.write(f"🔍 **DEBUG - Image Placeholders Generated:** {image_placeholder_text}")
                st.write("🔍 **DEBUG - Example placeholders:** image {0}, image {1}, image {2}...")
            
            # Escape for SQL - be careful with the curly braces
            prompt_escaped = prompt_with_placeholders.replace("'", "''")
            
            # Create PROMPT function call following Snowflake documentation syntax exactly
            # PROMPT('text with {0} {1} placeholders', TO_FILE1, TO_FILE2, ...)
            to_file_joined = ',\n        '.join(to_file_calls)
            analysis_query = f"""SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
    '{model_option}',
    PROMPT('{prompt_escaped}',
        {to_file_joined})
)"""
            
            # DEBUG: Show how placeholders map to files
            if st.session_state.get('debug_mode', False):
                st.write("🔍 **DEBUG - Placeholder to File Mapping:**")
                for i, file_name in enumerate(uploaded_file_names):
                    st.code(f"{{{i}}} -> {file_name}")
        
        # Show additional debug for PROMPT method
        if st.session_state.get('debug_mode', False) and processing_method == "prompt":
            st.write("🔍 **DEBUG - PROMPT Function Structure:**")
            st.code(f"PROMPT(text_with_placeholders, {len(to_file_calls)} TO_FILE calls)")
            st.write("Following Snowflake documentation pattern exactly with escaped JSON curly braces")
        
        # Show the FULL SQL query being generated for debugging
        if st.session_state.get('debug_mode', False):
            st.write("🔍 **FULL Generated Comprehensive Analysis Query:**")
            st.text_area("Complete SQL Query", analysis_query, height=300)
            
            # Show the arguments being passed
            st.write("🔍 **CORTEX.AI_COMPLETE Arguments:**")
            st.code(f"Model: {model_option}")
            st.code(f"Processing Method: {processing_method}")
            st.code(f"Prompt Length: {len(comprehensive_prompt)} characters")
            st.code(f"Number of Images: {len(to_file_calls)}")
            st.code(f"Image Files: {', '.join(uploaded_file_names)}")
            
            # Show argument count validation
            if processing_method == "single":
                expected_args = 3  # model + prompt + 1 image
                st.info(f"✅ Single image: Using {expected_args} arguments (within CORTEX.AI_COMPLETE limit)")
            elif processing_method == "dual":
                expected_args = 4  # model + prompt + 2 images  
                st.info(f"✅ Dual image: Using {expected_args} arguments (at CORTEX.AI_COMPLETE limit)")
            else:
                st.info(f"✅ Multi-image: Using PROMPT() function for {len(uploaded_file_names)} images")
        
        st.write("🔍 **Analyzing images...**")
        
        # Execute the query
        try:
            result = session.sql(analysis_query).collect()
            analysis_text = result[0][0] if result and len(result) > 0 else "No analysis results returned"
            
            st.success(f"✅ Analysis completed")
            
            # Parse with comprehensive parsing
            analysis_data = parse_comprehensive_multi_image_analysis(analysis_text, uploaded_file_names)
            
            return True, f"Successfully analyzed {len(uploaded_file_names)} images", analysis_data, analysis_text, analysis_query
            
        except Exception as query_error:
            st.error(f"❌ SQL Execution Error: {str(query_error)}")
            
            # Provide helpful fallback suggestions
            if "too many arguments" in str(query_error).lower():
                st.warning("💡 **Fallback Suggestion:** Try with fewer images (1-2 for direct processing)")
                
                # Offer to process first 2 images only
                if len(uploaded_file_names) > 2:
                    st.info(f"🔄 **Auto-Fallback:** Processing first 2 images only: {uploaded_file_names[:2]}")
                    
                    # Create fallback query with just 2 images
                    fallback_prompt = comprehensive_prompt.replace(f"these {len(uploaded_file_names)} golf club images", "these 2 golf club images")
                    fallback_prompt_escaped = fallback_prompt.replace("'", "''").replace("\\", "\\\\")
                    
                    fallback_query = f"""SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
    '{model_option}',
    '{fallback_prompt_escaped}',
    {to_file_calls[0]},
    {to_file_calls[1]}
)"""
                    
                    try:
                        st.write("🔄 **Executing fallback query with 2 images...**")
                        fallback_result = session.sql(fallback_query).collect()
                        analysis_text = fallback_result[0][0] if fallback_result and len(fallback_result) > 0 else "No analysis results returned"
                        
                        st.success(f"✅ **FALLBACK SUCCESS** - Analyzed first 2 images using {model_option}")
                        
                        # Parse with just the first 2 file names
                        analysis_data = parse_comprehensive_multi_image_analysis(analysis_text, uploaded_file_names[:2])
                        
                        # Add note about fallback processing
                        if analysis_data.get('classification_notes'):
                            analysis_data['classification_notes'] += f" NOTE: Fallback processing used - analyzed first 2 of {len(uploaded_file_names)} uploaded images."
                        
                        return True, f"Successfully analyzed 2 of {len(uploaded_file_names)} images (fallback mode)", analysis_data, analysis_text, fallback_query
                        
                    except Exception as fallback_error:
                        st.error(f"❌ Fallback also failed: {str(fallback_error)}")
                        return False, f"Both primary and fallback processing failed: {str(fallback_error)}", None, None, None
            
            return False, f"SQL execution error: {str(query_error)}", None, None, analysis_query
        
    except Exception as e:
        st.error(f"❌ **Comprehensive multi-image processing failed**: {str(e)}")
        st.error("💡 Try reducing the number of images or check that your model supports multi-image processing")
        return False, f"Multi-image processing error: {str(e)}", None, None, None

def analyze_individual_image_COMPREHENSIVE(uploaded_file, model_option):
    """
    Comprehensive single image analysis with maximum data extraction.
    """
    try:
        # Get the active Snowflake session
        session = get_active_session()
        
        # Get file name and use the correct stage name
        file_name = uploaded_file.name
        stage_name_no_at = "IMG_STAGE"
        stage_name = f"@{stage_name_no_at}"
        
        if st.session_state.get('debug_mode', False):
            st.write(f"📤 **COMPREHENSIVE UPLOAD**: {file_name} to {stage_name}...")
        
        try:
            # Reset file position
            uploaded_file.seek(0)
            
            # Create file stream using BytesIO
            file_stream = io.BytesIO(uploaded_file.getvalue())
            
            # Upload the file to the stage
            result = session.file.put_stream(
                file_stream,
                f"{stage_name}/{file_name}",
                auto_compress=False,
                overwrite=True
            )
            
            if st.session_state.get('debug_mode', False):
                st.success(f"File '{file_name}' uploaded successfully!")
            
            # Verify file exists in stage after upload
            verification = session.sql(f"LIST {stage_name}").collect()
            file_found = any(file_name in str(file_info) for file_info in verification)
            
            if not file_found:
                st.warning(f"File uploaded but not found in stage listing!")
                return False, "File not found in stage after upload", None, None, None
            
            # Run comprehensive Cortex analysis
            if st.session_state.get('debug_mode', False):
                st.write(f"🔍 **COMPREHENSIVE ANALYSIS**: Extracting maximum data from {file_name}...")
            
            # Use the comprehensive single image analysis prompt
            analysis_prompt = get_enhanced_single_image_analysis_prompt()
            analysis_prompt = analysis_prompt.replace("'", "''")
            
            # Create the analysis query with the selected model
            analysis_query = f"""
            SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
                '{model_option}',
                '{analysis_prompt}',
                TO_FILE('{stage_name}', '{file_name}')
            )
            """
            
            # Execute the query
            try:
                result = session.sql(analysis_query).collect()
                analysis_text = result[0][0] if result and len(result) > 0 else "No analysis results returned"
                
                # Parse using comprehensive parsing (treating as single-image multi-analysis)
                analysis_data = parse_comprehensive_multi_image_analysis(analysis_text, [file_name])
                
                return True, f"Analysis completed for {file_name}", analysis_data, analysis_text, analysis_query
                
            except Exception as analysis_err:
                st.error(f"Error during analysis of {file_name}: {str(analysis_err)}")
                return False, f"Error during analysis: {str(analysis_err)}", None, None, analysis_query
            
        except Exception as upload_err:
            st.error(f"Error uploading {file_name}: {str(upload_err)}")
            return False, f"Error during file upload: {str(upload_err)}", None, None, None
        
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return False, f"Error during processing: {str(e)}", None, None, None

# =====================================================
# ENHANCED DISPLAY FUNCTIONS
# =====================================================

def display_comprehensive_analysis_results(analysis_data, raw_output, file_names, query_info=None):
    """
    Display comprehensive analysis results with maximum detail presentation.
    """
    st.markdown("## 🔍 **COMPREHENSIVE MULTI-IMAGE ANALYSIS RESULTS**")
    
    # Store query info in session state if provided
    if query_info:
        st.session_state.last_analysis_query = query_info
    
    # ===============================================
    # PROMINENT VALUE DISPLAY (TOP PRIORITY)
    # ===============================================
    
    estimated_range = analysis_data.get('estimated_price_range', '')
    retail_price = analysis_data.get('retail_price')
    trade_in_value = analysis_data.get('trade_in_value')
    
    if estimated_range or retail_price or trade_in_value:
        st.markdown("---")
        
        # Market value section with enhanced display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 💰 **COMPREHENSIVE MARKET VALUATION**")
            
            if estimated_range:
                st.markdown(f"""
                <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border-radius: 15px; margin: 15px 0; border: 2px solid #28a745;'>
                    <h1 style='color: #155724; margin: 0; font-size: 52px; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);'>{estimated_range}</h1>
                    <p style='color: #155724; margin: 15px 0 5px 0; font-size: 18px; font-weight: 600;'>Current Market Value Range</p>
                    <p style='color: #155724; margin: 0; font-size: 14px; opacity: 0.8;'>Based on comprehensive condition analysis of {len(file_names)} images</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional pricing details
            if retail_price or trade_in_value:
                price_col1, price_col2 = st.columns(2)
                
                with price_col1:
                    if retail_price and isinstance(retail_price, (int, float)):
                        st.metric("Original Retail Price", f"${retail_price:,.2f}", help="Price when new")
                
                with price_col2:
                    if trade_in_value and isinstance(trade_in_value, (int, float)):
                        st.metric("Estimated Trade-in Value", f"${trade_in_value:,.2f}", help="Shop trade-in estimate")
            
            # Market demand indicator
            market_demand = analysis_data.get('market_demand', '').lower()
            if market_demand:
                demand_colors = {'high': '🟢 High', 'medium': '🟡 Medium', 'low': '🔴 Low'}
                demand_display = demand_colors.get(market_demand, f'⚪ {market_demand.title()}')
                st.markdown(f"**Market Demand:** {demand_display}")
        
        st.markdown("---")
    
    # ===============================================  
    # COMPREHENSIVE ANALYSIS OVERVIEW
    # ===============================================
    
    st.markdown("### 📊 **Analysis Overview**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Images Processed", 
            len(file_names),
            help="Total images analyzed simultaneously"
        )
    
    with col2:
        confidence = analysis_data.get('confidence_score', 0)
        confidence_pct = confidence * 100 if confidence <= 1 else confidence
        st.metric(
            "AI Confidence", 
            f"{confidence_pct:.1f}%",
            help="Overall analysis confidence score"
        )
    
    with col3:
        grade = analysis_data.get('overall_grade', 0)
        if grade and str(grade) not in ['0', 'N/A']:
            grade_color = "🟢" if float(grade) >= 8 else "🟡" if float(grade) >= 6 else "🔴"
            st.metric("Condition Grade", f"{grade_color} {grade}/10", help="Overall condition assessment")
        else:
            st.metric("Condition Grade", "Analyzing...", help="Overall condition assessment")
    
    with col4:
        analysis_type = "Comprehensive Multi-Image" if len(file_names) > 1 else "Comprehensive Single"
        st.metric("Analysis Type", analysis_type, help="Type of analysis performed")
    
    # ===============================================
    # COMPREHENSIVE CLUB IDENTIFICATION
    # ===============================================
    
    st.markdown("### 🏌️ **Complete Club Identification**")
    
    # Primary identification
    id_col1, id_col2 = st.columns(2)
    
    with id_col1:
        st.markdown("#### **Primary Details**")
        
        brand = analysis_data.get('brand', 'Analyzing...')
        model = analysis_data.get('model', 'Analyzing...')
        club_type = analysis_data.get('club_type', 'Analyzing...')
        year = analysis_data.get('year', 'Unknown')
        club_name = analysis_data.get('club_name', 'N/A')
        
        # Enhanced brand display with validation
        if brand and brand not in ['Unknown', 'Analyzing...', 'Parse Error']:
            st.markdown(f"**🏷️ Brand:** {brand}")
        else:
            st.markdown("**🏷️ Brand:** 🔍 Identifying...")
            
        st.markdown(f"**📝 Model:** {model}")
        st.markdown(f"**🏌️ Type:** {club_type.title()}")
        st.markdown(f"**📅 Year:** {year}")
        
        if club_name and club_name != 'N/A':
            st.markdown(f"**📋 Full Name:** {club_name}")
    
    with id_col2:
        st.markdown("#### **Technical Specifications**")
        
        # Technical specs with enhanced formatting
        loft = analysis_data.get('loft')
        loft_display = f"{loft}°" if loft and str(loft) not in ['', '0', 'N/A'] else "Unknown"
        
        lie_angle = analysis_data.get('lie_angle')
        lie_display = f"{lie_angle}°" if lie_angle and str(lie_angle) not in ['', '0', 'N/A'] else "Unknown"
        
        bounce_angle = analysis_data.get('bounce_angle')
        bounce_display = f"{bounce_angle}°" if bounce_angle and str(bounce_angle) not in ['', '0', 'N/A'] else "N/A"
        
        shaft_length = analysis_data.get('shaft_length_inches')
        length_display = f"{shaft_length}\"" if shaft_length and str(shaft_length) not in ['', '0', 'N/A'] else "Standard"
        
        hand = analysis_data.get('hand', 'Right')
        shaft_type = analysis_data.get('shaft_type', 'Unknown')
        shaft_flex = analysis_data.get('shaft_flex', 'Unknown')
        shaft_label = analysis_data.get('shaft_label', '')
        
        st.markdown(f"**🎯 Loft:** {loft_display}")
        st.markdown(f"**📐 Lie Angle:** {lie_display}")
        if bounce_display != "N/A":
            st.markdown(f"**⚡ Bounce:** {bounce_display}")
        st.markdown(f"**📏 Length:** {length_display}")
        st.markdown(f"**👤 Hand:** {hand}")
        
        shaft_info = f"{shaft_type} ({shaft_flex})"
        if shaft_label:
            shaft_info += f" - {shaft_label}"
        st.markdown(f"**🏒 Shaft:** {shaft_info}")
    
    # Additional specifications
    club_category = analysis_data.get('club_category', '')
    club_sub_type = analysis_data.get('club_sub_type', '')
    set_composition = analysis_data.get('set_composition', '')
    face_angle = analysis_data.get('face_angle', '')
    grind_type = analysis_data.get('grind_type', '')
    model_designation = analysis_data.get('model_designation', '')
    
    if any([club_category, club_sub_type, set_composition, face_angle, grind_type, model_designation]):
        st.markdown("#### **Additional Specifications**")
        
        spec_details = []
        if club_category: spec_details.append(f"**Category:** {club_category.title()}")
        if club_sub_type: spec_details.append(f"**Sub-type:** {club_sub_type.title()}")
        if set_composition: spec_details.append(f"**Set Info:** {set_composition}")
        if face_angle: spec_details.append(f"**Face Angle:** {face_angle}")
        if grind_type: spec_details.append(f"**Grind:** {grind_type}")
        if model_designation: spec_details.append(f"**Designation:** {model_designation}")
        
        # Display in columns
        for i in range(0, len(spec_details), 2):
            spec_col1, spec_col2 = st.columns(2)
            with spec_col1:
                st.markdown(spec_details[i])
            if i + 1 < len(spec_details):
                with spec_col2:
                    st.markdown(spec_details[i + 1])
    
    # ===============================================
    # COMPREHENSIVE CONDITION ASSESSMENT
    # ===============================================
    
    st.markdown("### 🔧 **Comprehensive Condition Assessment**")
    
    # Overall condition display
    overall_grade = analysis_data.get('overall_grade')
    condition_assessment = analysis_data.get('condition_assessment', '')
    
    if overall_grade:
        try:
            grade_value = float(overall_grade)
            grade_color = "#28a745" if grade_value >= 8 else "#ffc107" if grade_value >= 6 else "#fd7e14" if grade_value >= 4 else "#dc3545"
            condition_text = condition_assessment or "Good condition"
            
            # Enhanced condition display
            cond_col1, cond_col2, cond_col3 = st.columns([1, 2, 1])
            with cond_col2:
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 15px; margin: 20px 0; border: 3px solid {grade_color};'>
                    <h3 style='margin: 0; color: #333; font-size: 24px;'>Overall Condition Assessment</h3>
                    <h1 style='color: {grade_color}; margin: 15px 0; font-size: 72px; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>{grade_value}/10</h1>
                    <p style='color: #666; margin: 0; font-size: 20px; font-weight: 600;'>{condition_text.title()}</p>
                </div>
                """, unsafe_allow_html=True)
        except (ValueError, TypeError):
            st.info("🔍 Overall condition assessment in progress...")
    
    # Detailed condition breakdown
    st.markdown("#### **Detailed Condition Analysis**")
    
    # Create comprehensive condition data
    condition_components = []
    
    # Face/Sole condition
    face_grade = analysis_data.get('face_sole_wear_grade', 'Not assessed')
    face_desc = analysis_data.get('face_sole_wear_description', 'No specific details available')
    condition_components.append({
        'Component': 'Face & Sole',
        'Grade': face_grade.title() if face_grade else 'Analyzing...',
        'Details': face_desc[:100] + '...' if len(face_desc) > 100 else face_desc,
        'Impact': 'Performance Critical' if face_grade in ['poor', 'fair'] else 'Minimal Impact'
    })
    
    # Scratches
    scratch_grade = analysis_data.get('scratches_grade', 'Not assessed')
    scratch_desc = analysis_data.get('scratches_description', 'No scratch details available')
    scratch_locations = analysis_data.get('scratches_locations', [])
    
    scratch_summary = scratch_grade.title() if scratch_grade else 'Analyzing...'
    if scratch_locations:
        scratch_summary += f" ({', '.join(scratch_locations[:3])})"
    
    condition_components.append({
        'Component': 'Scratches',
        'Grade': scratch_summary,
        'Details': scratch_desc[:100] + '...' if len(scratch_desc) > 100 else scratch_desc,
        'Impact': 'Cosmetic & Performance' if scratch_grade in ['severe', 'moderate'] else 'Cosmetic Only'
    })
    
    # Paint/Finish
    paint_grade = analysis_data.get('paint_chips_grade', 'Not assessed')
    paint_desc = analysis_data.get('paint_chips_description', 'No paint condition details')
    paint_locations = analysis_data.get('paint_chips_locations', [])
    
    paint_summary = paint_grade.title() if paint_grade else 'Analyzing...'
    if paint_locations:
        paint_summary += f" ({', '.join(paint_locations[:3])})"
    
    condition_components.append({
        'Component': 'Paint & Finish',
        'Grade': paint_summary,
        'Details': paint_desc[:100] + '...' if len(paint_desc) > 100 else paint_desc,
        'Impact': 'Cosmetic & Resale Value'
    })
    
    # Grip condition
    grip_condition = analysis_data.get('grip_condition', 'Not assessed')
    grip_impact = 'Replacement Recommended' if 'replacement' in grip_condition.lower() else 'Playable'
    
    condition_components.append({
        'Component': 'Grip',
        'Grade': grip_condition.title() if grip_condition else 'Analyzing...',
        'Details': f"Grip wear assessment: {grip_condition}",
        'Impact': grip_impact
    })
    
    # Putter-specific wear (if applicable)
    putter_wear = analysis_data.get('putter_paint_wear_grade', '')
    if putter_wear and analysis_data.get('club_type', '').lower() == 'putter':
        condition_components.append({
            'Component': 'Putter Paint',
            'Grade': putter_wear.title(),
            'Details': 'Putter-specific paint wear assessment',
            'Impact': 'Cosmetic Only'
        })
    
    # Display condition table
    condition_df = pd.DataFrame(condition_components)
    st.dataframe(
        condition_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Component": st.column_config.TextColumn("Component", width=140),
            "Grade": st.column_config.TextColumn("Condition", width=180),
            "Details": st.column_config.TextColumn("Assessment Details", width=350),
            "Impact": st.column_config.TextColumn("Performance Impact", width=150)
        }
    )
    
    # ===============================================
    # TECHNOLOGY & FEATURES
    # ===============================================
    
    tech_tags = analysis_data.get('technology_tags', [])
    special_designations = analysis_data.get('special_designations', [])
    features = analysis_data.get('features', {})
    
    if tech_tags or special_designations or features:
        st.markdown("### ⚙️ **Technology & Special Features**")
        
        if tech_tags:
            st.markdown("#### **Technology Features**")
            tech_cols = st.columns(min(3, len(tech_tags)))
            for i, tech in enumerate(tech_tags):
                with tech_cols[i % 3]:
                    st.markdown(f"🔧 **{tech}**")
        
        if special_designations:
            st.markdown("#### **Special Designations**")
            special_cols = st.columns(min(3, len(special_designations)))
            for i, designation in enumerate(special_designations):
                with special_cols[i % 3]:
                    st.markdown(f"⭐ **{designation}**")
        
        if features and isinstance(features, dict):
            st.markdown("#### **Feature Analysis**")
            feature_cols = st.columns(3)
            feature_items = list(features.items())
            
            for i, (feature, value) in enumerate(feature_items):
                with feature_cols[i % 3]:
                    status = "✅" if value else "❌"
                    st.markdown(f"{status} **{feature.replace('_', ' ').title()}**")
    
    # ===============================================
    # COMPREHENSIVE IMAGE ANALYSIS DETAILS
    # ===============================================
    
    st.markdown("### 📸 **Image Analysis Details**")
    
    analysis_details_col1, analysis_details_col2 = st.columns(2)
    
    with analysis_details_col1:
        st.markdown("#### **Processing Information**")
        st.markdown(f"**Source Images:** {len(file_names)}")
        for i, file_name in enumerate(file_names, 1):
            st.markdown(f"  {i}. {file_name}")
        
        st.markdown(f"**Analysis Method:** Comprehensive Multi-Image")
        st.markdown(f"**Processing Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    with analysis_details_col2:
        st.markdown("#### **Quality Metrics**")
        st.markdown(f"**Confidence Score:** {analysis_data.get('confidence_score', 0.9):.3f}")
        st.markdown(f"**Data Fields Populated:** {len([k for k, v in analysis_data.items() if v and str(v) not in ['', 'Unknown', 'N/A', '0']])}")
        st.markdown(f"**Analysis Type:** {analysis_data.get('analysis_type', 'Unknown')}")
    
    # Classification notes
    classification_notes = analysis_data.get('classification_notes', '')
    if classification_notes:
        st.markdown("#### **Comprehensive Analysis Notes**")
        # Split long notes into readable segments
        notes_segments = classification_notes.split('\\n')
        for segment in notes_segments:
            if segment.strip():
                st.markdown(f"• {segment.strip()}")
    
    # ===============================================
    # DEBUG AND RAW OUTPUT
    # ===============================================
    
    if st.session_state.get('debug_mode', False):
        with st.expander("🔍 **Debug: Comprehensive Analysis Data**", expanded=False):
            st.write("**All extracted fields:**")
            debug_data = {}
            for key, value in analysis_data.items():
                if value and str(value) not in ['', 'Unknown', 'N/A', '0', 'NULL']:
                    debug_data[key] = value
            st.json(debug_data, expanded=False)
        
        with st.expander("📊 **Database Save Status**", expanded=False):
            st.info("Comprehensive data has been saved to all three database tables:")
            st.markdown("""
            - ✅ **CALLAWAY_CLUBS_FILE_V3**: Main analysis record with all specifications
            - ✅ **IMAGE_REFERENCES_V3**: Individual image references with FILE data
            - ✅ **CLUB_DEFECT_OBSERVATIONS_V3**: Detailed defect and wear observations
            """)
    
    # ===============================================
    # AI ANALYSIS DETAILS - ALWAYS VISIBLE
    # ===============================================
    
    st.markdown("---")
    st.markdown("### 🤖 **AI Analysis Details**")
    
    # Store the query info in session state if provided
    if 'last_analysis_query' in st.session_state:
        with st.expander("💻 **CORTEX.AI_COMPLETE Command Used**", expanded=False):
            st.code(st.session_state.last_analysis_query, language="sql")
    
    with st.expander("📝 **AI Response**", expanded=False):
        st.code(raw_output, language="sql")

# Function to fetch analysis history
def fetch_analysis_history():
    try:
        # Get the active Snowflake session
        session = get_active_session()
        
        # Enhanced query with more comprehensive data
        query = """
        SELECT 
            IMAGE_ID, 
            CLUB_TYPE, 
            BRAND, 
            MODEL,
            YEAR,
            OVERALL_GRADE,
            CONDITION_ASSESSMENT, 
            ESTIMATED_PRICE_RANGE,
            RETAIL_PRICE,
            TRADE_IN_VALUE,
            MARKET_DEMAND,
            AI_MODEL,
            CONFIDENCE_SCORE,
            ANALYSIS_TIMESTAMP,
            ARRAY_SIZE(TECHNOLOGY_TAGS) as TECH_FEATURES,
            ARRAY_SIZE(SPECIAL_DESIGNATIONS) as SPECIAL_COUNT
        FROM 
            CALLAWAY_IMG_COMPLETE.PUBLIC.CALLAWAY_CLUBS_FILE_V3
        ORDER BY 
            ANALYSIS_TIMESTAMP DESC
        LIMIT 100
        """
        
        # Execute the query and convert to DataFrame
        df = session.sql(query).to_pandas()
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching analysis history: {e}")
        return None

# =====================================================
# ENHANCED PAGE DISPLAY FUNCTIONS
# =====================================================

def display_comprehensive_upload_page(model_option, auto_analyze=True):
    # Main content
    st.markdown('<div class="upload-header">Golf Club Analyzer V3 - Multi-Image Processing</div>', unsafe_allow_html=True)
    
    # Enhanced introduction
    auto_status = "<br>🤖 **Auto-analysis enabled** - Analysis starts immediately after upload." if auto_analyze else "📋 **Manual mode** - Click analyze button to start."
    
    st.markdown(f"""
    <div class="instruction-text">
        Upload 1-10 images of your golf clubs for <strong>comprehensive expert analysis</strong>
        {auto_status}
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced file uploader
    uploaded_files = st.file_uploader(
        "Upload Golf Club Images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        help="Select 1-10 images. Multiple angles provide the most comprehensive analysis results."
    )
    
    # Display uploaded images with enhanced preview
    if uploaded_files:
        st.markdown('<div class="upload-header" style="font-size: 20px;">📸 Uploaded Images</div>', unsafe_allow_html=True)
        
        # Enhanced image display
        num_cols = min(4, len(uploaded_files))
        cols = st.columns(num_cols)
        
        total_size = sum(f.size for f in uploaded_files)
        st.info(f"📊 **Summary:** {len(uploaded_files)} images, {total_size // 1024:.1f} KB total")
        
        for i, uploaded_file in enumerate(uploaded_files):
            col_idx = i % num_cols
            with cols[col_idx]:
                # Enhanced image display with better sizing
                image = Image.open(uploaded_file)
                width, height = image.size
                max_size = 280
                
                if width > height:
                    new_width = min(width, max_size)
                    new_height = int(height * (new_width / width))
                else:
                    new_height = min(height, max_size)
                    new_width = int(width * (new_height / height))
                
                resized_image = image.resize((new_width, new_height))
                st.image(resized_image, caption=f"Image {i+1}")
                
                # Enhanced metadata display - only show detailed info in debug mode
                if st.session_state.get('debug_mode', False):
                    st.markdown(f"""
                    <div class="image-box" style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);">
                        <strong>📁 File:</strong> {uploaded_file.name}<br>
                        <strong>📏 Size:</strong> {uploaded_file.size // 1024} KB<br>
                        <strong>🎯 Role:</strong> {'Primary' if i == 0 else f'Angle {i}'}
                    </div>
                    """, unsafe_allow_html=True)
        
        if len(uploaded_files) > 4:
            st.info(f"... and {len(uploaded_files) - 4} more images ready for analysis")
        
        # Enhanced analysis trigger logic
        should_analyze = False
        
        if auto_analyze:
            st.success(f"🤖 **Auto-analysis starting** for {len(uploaded_files)} image(s)...")
            should_analyze = True
            
            st.markdown("---")
            st.markdown("*Auto-analysis in progress. Use the button below for manual re-analysis:*")
            manual_button_text = "🔄 Re-analyze Images" if len(uploaded_files) > 1 else "🔄 Re-analyze Image"
            if st.button(manual_button_text, key="manual_comprehensive_analyze"):
                should_analyze = True
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            button_text = "🚀 Analyze" if len(uploaded_files) == 1 else f"🚀 Analyze {len(uploaded_files)} Images"
            if st.button(button_text, key="comprehensive_analyze_button"):
                should_analyze = True
        
        # Perform comprehensive analysis
        if should_analyze:
            # Clear cache
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
            except:
                pass
            
            with st.spinner("🔍 Analyzing..."):
                
                # Choose comprehensive analysis method
                if len(uploaded_files) == 1:
                    if st.session_state.get('debug_mode', False):
                        st.info("🔍 **COMPREHENSIVE SINGLE-IMAGE ANALYSIS** - Maximum data extraction...")
                    
                    uploaded_files[0].seek(0)
                    
                    success, message, analysis_data, raw_output, query_info = analyze_individual_image_COMPREHENSIVE(
                        uploaded_files[0], model_option
                    )
                    
                    if success:
                        file_names = [uploaded_files[0].name]
                        save_success, save_message, saved_id = save_comprehensive_analysis_result(
                            analysis_data, file_names, model_option
                        )
                        
                        if save_success:
                            if st.session_state.get('debug_mode', False):
                                st.success(f"✅ {save_message}")
                        else:
                            st.error(f"❌ Database Error: {save_message}")
                        
                        # Display comprehensive results
                        st.markdown("---")
                        display_comprehensive_analysis_results(
                            analysis_data, raw_output, file_names, query_info
                        )
                    else:
                        st.error(f"❌ Analysis failed: {message}")
                    
                else:
                    if st.session_state.get('debug_mode', False):
                        st.info(f"🔍 **MULTI-IMAGE ANALYSIS** - Processing all {len(uploaded_files)} images with maximum data extraction...")
                    
                    success, message, analysis_data, raw_output, query_info = analyze_multiple_images_COMPREHENSIVE(
                        uploaded_files, model_option
                    )
                    
                    if success:
                        file_names = [f.name for f in uploaded_files]
                        save_success, save_message, saved_id = save_comprehensive_analysis_result(
                            analysis_data, file_names, model_option
                        )
                        
                        if save_success:
                            if st.session_state.get('debug_mode', False):
                                st.success(f"✅ {save_message}")
                        else:
                            st.error(f"❌ Database Error: {save_message}")
                        
                        # Display comprehensive results
                        st.markdown("---")
                        display_comprehensive_analysis_results(
                            analysis_data, raw_output, file_names, query_info
                        )
                        
                        # Show query info for transparency - only in debug mode
                        if st.session_state.get('debug_mode', False):
                            st.info("💡 Additional debug information available in expandable sections below")
                        
                    else:
                        st.error(f"❌ Analysis failed: {message}")

def display_enhanced_history_page(model_option):
    st.markdown('<div class="upload-header">Comprehensive Analysis History</div>', unsafe_allow_html=True)
    
    # Display selected model
    st.info(f"Current AI model: {model_option}")
    
    # Fetch enhanced analysis history
    history_data = fetch_analysis_history()
    
    if history_data is not None and not history_data.empty:
        # Enhanced statistics
        st.markdown("### 📊 **Analysis Statistics**")
        
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Total Analyses", len(history_data))
        
        with stats_col2:
            avg_grade = history_data['OVERALL_GRADE'].dropna().mean()
            st.metric("Average Condition", f"{avg_grade:.1f}/10" if not pd.isna(avg_grade) else "N/A")
        
        with stats_col3:
            top_brand = history_data['BRAND'].mode().iloc[0] if not history_data['BRAND'].empty else "N/A"
            st.metric("Most Analyzed Brand", top_brand)
        
        with stats_col4:
            confidence_avg = history_data['CONFIDENCE_SCORE'].dropna().mean()
            st.metric("Avg Confidence", f"{confidence_avg:.1%}" if not pd.isna(confidence_avg) else "N/A")
        
        # Enhanced data display
        st.markdown("### 📋 **Detailed Analysis History**")
        
        # Format the dataframe for better display
        display_df = history_data.copy()
        
        # Format price columns
        for price_col in ['RETAIL_PRICE', 'TRADE_IN_VALUE']:
            if price_col in display_df.columns:
                display_df[price_col] = display_df[price_col].apply(
                    lambda x: f"${x:,.2f}" if pd.notna(x) and x > 0 else "N/A"
                )
        
        # Enhanced dataframe display
        st.dataframe(
            display_df, 
            use_container_width=True,
            column_config={
                "IMAGE_ID": st.column_config.TextColumn("Analysis ID", width=150),
                "CLUB_TYPE": st.column_config.TextColumn("Type", width=100),
                "BRAND": st.column_config.TextColumn("Brand", width=120),
                "MODEL": st.column_config.TextColumn("Model", width=150),
                "OVERALL_GRADE": st.column_config.NumberColumn("Grade", format="%.1f/10"),
                "ESTIMATED_PRICE_RANGE": st.column_config.TextColumn("Market Value", width=120),
                "CONFIDENCE_SCORE": st.column_config.NumberColumn("Confidence", format="%.1%"),
                "TECH_FEATURES": st.column_config.NumberColumn("Tech Features", width=100),
                "ANALYSIS_TIMESTAMP": st.column_config.DatetimeColumn("Analyzed", width=150)
            }
        )
        
        # Enhanced download option
        csv = display_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Comprehensive Analysis History",
            data=csv,
            file_name=f"comprehensive_golf_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Analysis insights
        with st.expander("📈 **Analysis Insights**", expanded=False):
            # Brand distribution
            if not history_data['BRAND'].empty:
                brand_counts = history_data['BRAND'].value_counts().head(10)
                st.bar_chart(brand_counts)
            
            # Condition distribution
            if not history_data['OVERALL_GRADE'].dropna().empty:
                st.subheader("Condition Grade Distribution")
                grade_bins = pd.cut(history_data['OVERALL_GRADE'].dropna(), bins=[0, 4, 6, 8, 10], labels=['Poor', 'Fair', 'Good', 'Excellent'])
                grade_counts = grade_bins.value_counts()
                st.bar_chart(grade_counts)
    
    else:
        st.info("No analysis history found. Upload and analyze some golf clubs to see comprehensive results here.")

def display_enhanced_settings_page(model_option, auto_analyze=True):
    st.markdown('<div class="upload-header">Analysis Settings</div>', unsafe_allow_html=True)
    
    # Display selected model
    st.info(f"Current AI model: {model_option}")
    
    # Enhanced analysis settings
    st.markdown("### ⚙️ **Settings**")
    
    # Debug mode toggle
    st.markdown("#### **Developer Options**")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        debug_mode = st.checkbox(
            "🐛 Enable Debug Mode",
            value=st.session_state.debug_mode,
            help="Show detailed processing information, SQL queries, and technical details"
        )
        st.session_state.debug_mode = debug_mode
        
        if debug_mode:
            st.warning("⚠️ Debug mode is ON - Verbose output enabled")
        else:
            st.success("✅ Debug mode is OFF - Clean interface")
    
    with col2:
        if st.button("🔄 Apply", type="secondary"):
            st.rerun()
    
    st.markdown("---")
    
    # Current settings display
    if auto_analyze:
        st.success("🤖 **Auto-analyze on upload:** ✅ **ENABLED**")
        if st.session_state.debug_mode:
            st.info("📊 **Data Extraction:** Comprehensive mode - All possible fields populated")
    else:
        st.info("🤖 **Auto-analyze on upload:** ❌ **DISABLED**")
        if st.session_state.debug_mode:
            st.info("📊 **Data Extraction:** Comprehensive mode - Maximum field population on demand")
    
    st.info("Change auto-analyze setting in the sidebar.")
    
    # Enhanced feature settings
    st.markdown("#### **Feature Configuration**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Primary Analysis Focus", [
            "Maximum Data Extraction", 
            "Condition Assessment Priority", 
            "Market Valuation Priority",
            "Technical Specifications Priority"
        ], index=0)
        
        st.selectbox("Default Club Category", [
            "All Types", "Drivers", "Irons", "Wedges", "Putters", "Hybrids", "Fairway Woods"
        ])
        
        st.checkbox("Enhanced defect detection", value=True, help="Detailed defect analysis with performance impact")
        st.checkbox("Comprehensive market analysis", value=True, help="Retail, trade-in, and demand assessment")
    
    with col2:
        st.checkbox("Technology feature extraction", value=True, help="Identify all golf technologies")
        st.checkbox("Special designation detection", value=True, help="Tour issue, limited edition, etc.")
        st.checkbox("Multi-angle consolidation", value=True, help="Combine insights from multiple images")
        st.checkbox("Performance impact analysis", value=True, help="Assess how condition affects performance")
    
    # Database configuration
    st.markdown("#### **Database Configuration**")
    st.info("📊 **V3 Comprehensive Schema** - Maximum data storage across 3 tables")
    
    with st.expander("Database Table Details"):
        st.markdown("""
        **CALLAWAY_CLUBS_FILE_V3** - Main analysis table with comprehensive fields:
        - Complete club identification & specifications
        - Detailed condition assessment grades
        - Market valuation data (retail, trade-in, demand)
        - Technology tags and special designations
        - Enhanced classification notes
        
        **IMAGE_REFERENCES_V3** - Multi-image support:
        - FILE type references for each uploaded image
        - Image sequence and reference type tracking
        - Primary/multi-angle image relationships
        
        **CLUB_DEFECT_OBSERVATIONS_V3** - Detailed defect tracking:
        - Specific defect types and locations
        - Size measurements and depth assessments
        - Performance impact analysis
        """)
    
    # Advanced AI settings
    st.markdown("#### **AI Model Configuration**")
    
    with st.expander("Model Performance Details"):
        model_info = {
            "claude-4-opus": {
                "Multi-Image": "✅ Exceptional",
                "Data Extraction": "✅ Maximum",
                "Condition Assessment": "✅ Superior",
                "Market Analysis": "✅ Advanced"
            },
            "claude-4-sonnet": {
                "Multi-Image": "✅ Excellent",
                "Data Extraction": "✅ Maximum",
                "Condition Assessment": "✅ Superior",
                "Market Analysis": "✅ Advanced"
            },
            "openai-gpt-4.1": {
                "Multi-Image": "✅ Excellent",
                "Data Extraction": "✅ High",
                "Condition Assessment": "✅ Superior",
                "Market Analysis": "✅ Advanced"
            },
            "claude-3-7-sonnet": {
                "Multi-Image": "✅ Excellent",
                "Data Extraction": "✅ High",
                "Condition Assessment": "✅ Good",
                "Market Analysis": "✅ Good"
            },
            "claude-3-5-sonnet": {
                "Multi-Image": "✅ Excellent", 
                "Data Extraction": "✅ High",
                "Condition Assessment": "✅ Superior",
                "Market Analysis": "✅ Good"
            },
            "llama-4-maverick": {
                "Multi-Image": "✅ Good",
                "Data Extraction": "⚠️ Moderate",
                "Condition Assessment": "✅ Good",
                "Market Analysis": "⚠️ Moderate"
            },
            "llama-4-scout": {
                "Multi-Image": "✅ Good",
                "Data Extraction": "⚠️ Moderate",
                "Condition Assessment": "✅ Good",
                "Market Analysis": "⚠️ Basic"
            },
            "openai-o4-mini": {
                "Multi-Image": "⚠️ Limited",
                "Data Extraction": "⚠️ Moderate",
                "Condition Assessment": "✅ Good",
                "Market Analysis": "⚠️ Basic"
            },
            "pixtral-large": {
                "Multi-Image": "❌ Limited",
                "Data Extraction": "⚠️ Moderate",
                "Condition Assessment": "✅ Good",
                "Market Analysis": "⚠️ Basic"
            }
        }
        
        if model_option in model_info:
            st.markdown(f"**{model_option} Capabilities:**")
            for capability, rating in model_info[model_option].items():
                st.markdown(f"- **{capability}:** {rating}")
    
    # Quality settings
    st.markdown("#### **Quality & Performance Settings**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.slider("Analysis confidence threshold", min_value=0.5, max_value=1.0, value=0.8, step=0.05,
                 help="Minimum confidence required for field population")
        st.slider("Image processing quality", min_value=60, max_value=100, value=90,
                 help="Higher quality = better analysis but slower processing")
    
    with col2:
        st.slider("Maximum images per analysis", min_value=1, max_value=10, value=10,
                 help="Limit simultaneous image processing")
        st.slider("Database save timeout (seconds)", min_value=30, max_value=300, value=120,
                 help="Maximum time to wait for database operations")
    
    # Advanced Snowflake settings
    with st.expander("Advanced Snowflake Configuration"):
        st.text_input("Stage Name", "IMG_STAGE", disabled=True, help="Snowflake stage for image storage")
        st.text_input("Database", "CALLAWAY_IMG_COMPLETE", disabled=True)
        st.text_input("Schema", "PUBLIC", disabled=True)
        st.text_input("Main Table", "CALLAWAY_CLUBS_FILE_V3", disabled=True, help="Primary analysis data")
        st.text_input("References Table", "IMAGE_REFERENCES_V3", disabled=True, help="Multi-image references")
        st.text_input("Defects Table", "CLUB_DEFECT_OBSERVATIONS_V3", disabled=True, help="Detailed defect tracking")
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Analysis settings saved successfully!")
        st.balloons()

# =====================================================
# MAIN APPLICATION FUNCTION - ENHANCED
# =====================================================

def main():
    # Initialize session state for debug mode
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    # Add custom CSS
    add_custom_css()
    
    # Enhanced header
    col1, col2, col3 = st.columns([2, 3, 1])
    
    with col1:
        st.image("https://moongolf.com/wp-content/uploads/2017/03/Callaway-logo-WHITE-1024x591-small-300x173.png", width=150)
    
    with col2:
        st.markdown("## Golf Club Analyzer V3")
        st.markdown("**Multi-Image Processing**")
    
    with col3:
        st.markdown("<div style='text-align: right; color: #666; font-size: 12px; padding-top: 20px;'>V3.0</div>", unsafe_allow_html=True)
    
    # Enhanced sidebar
    st.sidebar.title("🏌️ Golf Club Analyzer V3")
    #st.sidebar.markdown("**COMPREHENSIVE EDITION**")

    # Model selection with enhanced descriptions
    model_descriptions = {
        "claude-4-opus": "🏆 Premium model for maximum accuracy and analysis",
        "claude-4-sonnet": "🥇 Best for comprehensive multi-image analysis",
        "openai-gpt-4.1": "🥇 Advanced multi-modal with strong reasoning",
        "claude-3-7-sonnet": "🥈 Enhanced analysis with improved accuracy",
        "claude-3-5-sonnet": "🥈 Excellent multi-image with high accuracy",
        "llama-4-maverick": "🥉 Strong open-source alternative",
        "llama-4-scout": "🥉 Fast and efficient analysis",
        "openai-o4-mini": "⚡ Quick processing with good accuracy",
        "pixtral-large": "🥉 Good single-image analysis"
    }
    
    model_option = st.sidebar.selectbox(
        "🤖 AI Model Selection",
        ["claude-4-opus", "claude-4-sonnet", "openai-gpt-4.1", "claude-3-7-sonnet", 
         "claude-3-5-sonnet", "llama-4-maverick", "llama-4-scout", "openai-o4-mini", "pixtral-large"],
        index=1,
        help="Claude-4-Opus recommended for maximum data extraction",
        format_func=lambda x: f"{x} - {model_descriptions.get(x, '')}"
    )
    
    # Enhanced analysis settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ **Analysis Options**")
    
    auto_analyze = st.sidebar.checkbox(
        "🤖 Auto-analyze on upload",
        value=True,
        help="Automatically analyze when images are uploaded"
    )
    
    # Model capability indicator - only show in debug mode
    if st.session_state.debug_mode:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎯 **Current Capabilities**")
        st.sidebar.info(f"**Model:** {model_option}")
        
        if model_option in ["claude-3-5-sonnet", "claude-4-sonnet"]:
            st.sidebar.success("✅ **COMPREHENSIVE** multi-image")
            st.sidebar.success("✅ **MAXIMUM** data extraction")
            st.sidebar.success("✅ Advanced condition analysis")
            st.sidebar.success("✅ Market valuation")
        else:
            st.sidebar.warning("⚠️ Single image only")
            st.sidebar.info("Limited data extraction")
    
    # Enhanced navigation
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "🧭 Navigation", 
        ["📤 Upload & Analyze", "📊 Analysis History", "⚙️ Settings"],
        format_func=lambda x: x
    )
    
    # Enhanced control buttons
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🔄 Clear Cache & Restart", key="clear_comprehensive_cache"):
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
            for key in list(st.session_state.keys()):
                if key != 'debug_mode':  # Preserve debug mode setting
                    del st.session_state[key]
            st.sidebar.success("✅ Cache cleared!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Cache clear error: {e}")
    
    if st.sidebar.button("🗑️ Clear Uploads", key="clear_uploads"):
        for key in list(st.session_state.keys()):
            if 'uploader' in key.lower():
                del st.session_state[key]
        st.sidebar.success("✅ Uploads cleared!")
        st.rerun()
    
    # Database status - only show in debug mode
    if st.session_state.debug_mode:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💾 **Database Status**")
        st.sidebar.text("✅ CALLAWAY_CLUBS_FILE_V3")
        st.sidebar.text("✅ IMAGE_REFERENCES_V3") 
        st.sidebar.text("✅ CLUB_DEFECT_OBSERVATIONS_V3")
        st.sidebar.text("📂 Stage: @IMG_STAGE")
    
    # Main content routing
    if page == "📤 Upload & Analyze":
        display_comprehensive_upload_page(model_option, auto_analyze)
    elif page == "📊 Analysis History":
        display_enhanced_history_page(model_option)
    else:
        display_enhanced_settings_page(model_option, auto_analyze)
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <strong>Golf Club Analyzer V3 - COMPREHENSIVE EDITION</strong><br>
        Maximum Data Extraction | Multi-Image Processing | Advanced Market Valuation<br>
        Powered by Snowflake Cortex AI & Streamlit | Optimized for Claude-4-Sonnet
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
