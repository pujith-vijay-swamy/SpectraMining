"""
SpectraMining - Satellite-based AI System for Mineral Resource Monitoring
A Streamlit dashboard for detecting illegal mining and predicting mineral deposits
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import ee
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Mock permitted mining sites database
PERMITTED_SITES = [
    {"name": "Eagle Mountain Mine", "lat": 33.7490, "lon": -115.4794, "permit_id": "CA-2023-001"},
    {"name": "Silver Peak Lithium", "lat": 37.7749, "lon": -117.6321, "permit_id": "NV-2022-045"},
]

# Spectral indices thresholds for mineral detection
IRON_OXIDE_THRESHOLD = 0.15
CLAY_CARBONATE_THRESHOLD = 0.12

# ============================================================================
# GOOGLE EARTH ENGINE SETUP
# ============================================================================

def initialize_gee():
    """Initialize Google Earth Engine"""
    try:
        ee.Initialize()
        return True
    except Exception as e:
        st.sidebar.warning(f"⚠️ GEE not authenticated. Using mock data mode.")
        with st.sidebar.expander("📖 How to enable real Sentinel-2 data"):
            st.markdown("""
            **Step 1:** Set up Google Cloud Project
            - Visit [Google Cloud Console](https://console.cloud.google.com/)
            - Create a new project
            - Enable Earth Engine API
            
            **Step 2:** Authenticate locally
            ```bash
            pip install google-cloud-sdk
            gcloud auth application-default login
            earthengine authenticate
            ```
            
            **Step 3:** Restart the Streamlit app
            
            Once authenticated, the app will automatically use real Sentinel-2 data.
            """)
        return False

# ============================================================================
# SATELLITE DATA FETCHER (Google Earth Engine)
# ============================================================================

def fetch_sentinel2_data(lat, lon, start_date, end_date, use_gee=False):
    """
    Fetch Sentinel-2 multispectral imagery from Google Earth Engine
    
    Args:
        lat, lon: Geographic coordinates
        start_date, end_date: Date range for imagery
        use_gee: Whether to use real GEE data or mock data
    
    Returns:
        Dictionary with spectral bands and calculated indices
    """
    
    if use_gee:
        try:
            # Define area of interest (1km radius)
            point = ee.Geometry.Point([lon, lat])
            aoi = point.buffer(1000)
            
            # Load Sentinel-2 surface reflectance data
            sentinel = ee.ImageCollection('COPERNICUS/S2_SR') \
                .filterBounds(aoi) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                .sort('system:time_start', False)  # Sort by most recent first
            
            # Check if data is available
            if sentinel.size().getInfo() == 0:
                st.warning("⚠️ No Sentinel-2 imagery available for this location and date range")
                return None
            
            # Get the most recent image
            image = sentinel.first().clip(aoi)
            
            # Select bands (B2=Blue, B3=Green, B4=Red, B8=NIR, B11=SWIR1, B12=SWIR2)
            bands = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
            
            # Sample the data at the point
            sample = bands.sampleRegions(
                collection=ee.FeatureCollection([ee.Feature(point)]),
                scale=10
            ).first().getInfo()
            
            if sample is None:
                st.warning("⚠️ Could not retrieve data for this location")
                return None
            
            props = sample['properties']
            b2_val = props.get('B2', 1000)
            b4_val = props.get('B4', 1000)
            b11_val = props.get('B11', 2500)
            b12_val = props.get('B12', 2000)
            
            return {
                'B2': props.get('B2', 1000),
                'B3': props.get('B3', 1000),
                'B4': b4_val,
                'B8': props.get('B8', 2500),
                'B11': b11_val,
                'B12': b12_val,
                'iron_oxide_index': b4_val / b2_val if b2_val > 0 else 0.15,
                'clay_carbonate_index': b11_val / b12_val if b12_val > 0 else 0.12,
                'data_source': 'Sentinel-2 (Real-time)',
                'acquisition_date': image.get('system:time_start').getInfo()
            }
        except Exception as e:
            st.warning(f"⚠️ Error fetching real GEE data: {str(e)}")
            st.info("Falling back to mock data mode for demonstration")
            return None
    else:
        # Mock data for testing without GEE authentication
        return {
            'B2': np.random.randint(800, 1500),
            'B3': np.random.randint(900, 1600),
            'B4': np.random.randint(1000, 1800),
            'B8': np.random.randint(2000, 3500),
            'B11': np.random.randint(2500, 4000),
            'B12': np.random.randint(2000, 3500),
            'iron_oxide_index': np.random.uniform(0.1, 0.3),
            'clay_carbonate_index': np.random.uniform(0.08, 0.18),
            'data_source': 'Mock Data (Dev Mode)',
            'acquisition_date': None
        }

# ============================================================================
# AI CLASSIFIER (Mock Version - Replace with trained model)
# ============================================================================

def classify_mining_activity(spectral_data):
    """
    Mock AI classifier - predicts if area contains mining activity
    Replace this with your trained CNN model
    
    Args:
        spectral_data: Dictionary with spectral bands and indices
    
    Returns:
        Dictionary with prediction and confidence
    """
    
    # Mock logic: Use spectral indices to simulate classification
    iron_score = spectral_data['iron_oxide_index'] / IRON_OXIDE_THRESHOLD
    clay_score = spectral_data['clay_carbonate_index'] / CLAY_CARBONATE_THRESHOLD
    
    # Combined anomaly score
    anomaly_score = (iron_score + clay_score) / 2
    
    # Simulate CNN output
    is_mining = anomaly_score > 1.0
    confidence = min(anomaly_score * 0.45 + 0.5, 0.99)  # Scale to 50-99% confidence
    
    return {
        'prediction': 'Mining' if is_mining else 'Natural',
        'confidence': confidence,
        'anomaly_score': anomaly_score
    }

# ============================================================================
# PREDICTIVE MINERAL MAPPING
# ============================================================================

def predict_mineral_deposits(spectral_data):
    """
    Predict potential mineral deposits using spectral anomalies
    This uses geological pattern recognition based on spectral signatures
    
    Returns:
        Dictionary with mineral probability and type
    """
    
    iron_index = spectral_data['iron_oxide_index']
    clay_index = spectral_data['clay_carbonate_index']
    
    # Iron ore deposits show high iron oxide signature
    iron_probability = min((iron_index / IRON_OXIDE_THRESHOLD) * 0.6, 0.95)
    
    # Clay/carbonate minerals indicate potential gold, copper, or lithium
    lithium_probability = min((clay_index / CLAY_CARBONATE_THRESHOLD) * 0.5, 0.90)
    
    # Combined mineral potential
    predictions = []
    
    if iron_probability > 0.65:
        predictions.append({
            'mineral': 'Iron Ore',
            'probability': iron_probability,
            'indicator': 'High Iron Oxide Signature'
        })
    
    if lithium_probability > 0.60:
        predictions.append({
            'mineral': 'Lithium/Rare Earth',
            'probability': lithium_probability,
            'indicator': 'Clay-Carbonate Anomaly'
        })
    
    if not predictions:
        predictions.append({
            'mineral': 'No Significant Deposits',
            'probability': 0.20,
            'indicator': 'Normal Spectral Pattern'
        })
    
    return predictions

# ============================================================================
# COMPLIANCE & LEGAL FLAGGING
# ============================================================================

def check_legal_status(lat, lon, is_mining):
    """
    Cross-reference detected mining with permit database
    Triggers RED ALERT if mining detected without permit
    
    Returns:
        Dictionary with legal status and alert level
    """
    
    if not is_mining:
        return {
            'status': 'N/A - No Mining Detected',
            'alert_level': 'GREEN',
            'permitted': None
        }
    
    # Check if coordinates are within permitted sites (simple distance check)
    for site in PERMITTED_SITES:
        distance = np.sqrt((lat - site['lat'])**2 + (lon - site['lon'])**2)
        if distance < 0.05:  # Within ~5km radius
            return {
                'status': f"✅ Permitted Site: {site['name']}",
                'alert_level': 'GREEN',
                'permitted': True,
                'permit_id': site['permit_id']
            }
    
    # Mining detected but no permit found - RED ALERT
    return {
        'status': '🚨 ILLEGAL MINING SUSPECTED',
        'alert_level': 'RED',
        'permitted': False,
        'permit_id': None
    }

# ============================================================================
# EVIDENCE PACKAGE GENERATOR
# ============================================================================

def generate_evidence_report(lat, lon, spectral_data, classification, legal_status):
    """
    Generate comprehensive evidence package for violations
    Includes GPS coordinates, timestamps, and historical activity
    """
    
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'coordinates': {
            'latitude': lat,
            'longitude': lon,
            'elevation': 'N/A'  # Can be added with DEM data
        },
        'detection': {
            'classification': classification['prediction'],
            'confidence': f"{classification['confidence']*100:.1f}%",
            'anomaly_score': f"{classification['anomaly_score']:.2f}"
        },
        'spectral_evidence': {
            'iron_oxide_index': f"{spectral_data['iron_oxide_index']:.3f}",
            'clay_carbonate_index': f"{spectral_data['clay_carbonate_index']:.3f}",
            'threshold_exceeded': spectral_data['iron_oxide_index'] > IRON_OXIDE_THRESHOLD
        },
        'legal_status': legal_status,
        'historical_activity': generate_mock_timeline(lat, lon),
        'recommended_action': 'Immediate field investigation required' if legal_status['alert_level'] == 'RED' else 'Continue monitoring'
    }
    
    return report

def generate_mock_timeline(lat, lon):
    """Generate mock historical activity timeline"""
    timeline = []
    for i in range(5):
        days_ago = (i + 1) * 30
        date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        activity = np.random.choice(['No Change', 'Vegetation Loss', 'Soil Disturbance', 'Equipment Detected'])
        timeline.append({'date': date, 'observation': activity})
    return timeline

# ============================================================================
# STREAMLIT DASHBOARD UI
# ============================================================================

def main():
    st.set_page_config(page_title="SpectraMining Dashboard", page_icon="🛰️", layout="wide")
    
    # Header
    st.title("🛰️ SpectraMining - AI Mineral Monitoring System")
    st.markdown("*Detect illegal mining • Predict mineral deposits • Protect natural resources*")
    
    # Initialize GEE
    gee_available = initialize_gee()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Monitoring Controls")
    
    # Location input
    st.sidebar.subheader("📍 Target Location")
    lat = st.sidebar.number_input("Latitude", value=33.7490, format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=-115.4794, format="%.4f")
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    st.sidebar.subheader("📅 Analysis Period")
    st.sidebar.text(f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Run analysis button
    if st.sidebar.button("🔍 Run Analysis", type="primary"):
        
        with st.spinner("🛰️ Acquiring satellite imagery..."):
            # STEP 1: ACQUIRE - Fetch Sentinel-2 data
            spectral_data = fetch_sentinel2_data(
                lat, lon, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d'),
                use_gee=gee_available
            )
        
        if spectral_data:
            # STEP 2: ANALYZE - AI Classification
            with st.spinner("🤖 Running AI classification..."):
                classification = classify_mining_activity(spectral_data)
            
            # STEP 3: VERIFY - Legal status check
            legal_status = check_legal_status(lat, lon, classification['prediction'] == 'Mining')
            
            # STEP 4: PREDICT - Mineral deposit prediction
            mineral_predictions = predict_mineral_deposits(spectral_data)
            
            # Display data source info
            st.info(f"📡 **Data Source:** {spectral_data.get('data_source', 'Unknown')}")
            if spectral_data.get('acquisition_date'):
                try:
                    from datetime import datetime as dt
                    acq_date = dt.fromtimestamp(spectral_data['acquisition_date'] / 1000)
                    st.caption(f"Acquisition Date: {acq_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                except:
                    pass
            
            # ================================================================
            # DISPLAY RESULTS
            # ================================================================
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "AI Classification",
                    classification['prediction'],
                    f"{classification['confidence']*100:.1f}% confidence"
                )
            
            with col2:
                alert_emoji = "🚨" if legal_status['alert_level'] == 'RED' else "✅"
                st.metric(
                    "Legal Status",
                    legal_status['alert_level'],
                    legal_status['status']
                )
            
            with col3:
                top_mineral = mineral_predictions[0]
                st.metric(
                    "Mineral Potential",
                    top_mineral['mineral'],
                    f"{top_mineral['probability']*100:.0f}% probability"
                )
            
            # Tabs for detailed information
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Spectral Analysis", "🗺️ Map View", "🔮 Mineral Predictions", "📋 Evidence Report"])
            
            with tab1:
                st.subheader("Spectral Indices")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric(
                        "Iron Oxide Index",
                        f"{spectral_data['iron_oxide_index']:.3f}",
                        f"Threshold: {IRON_OXIDE_THRESHOLD}"
                    )
                    if spectral_data['iron_oxide_index'] > IRON_OXIDE_THRESHOLD:
                        st.warning("⚠️ Iron oxide anomaly detected")
                
                with col_b:
                    st.metric(
                        "Clay/Carbonate Index",
                        f"{spectral_data['clay_carbonate_index']:.3f}",
                        f"Threshold: {CLAY_CARBONATE_THRESHOLD}"
                    )
                    if spectral_data['clay_carbonate_index'] > CLAY_CARBONATE_THRESHOLD:
                        st.warning("⚠️ Clay/carbonate anomaly detected")
                
                # Show spectral bands
                st.subheader("Raw Spectral Bands")
                bands_df = pd.DataFrame({
                    'Band': ['B2 (Blue)', 'B3 (Green)', 'B4 (Red)', 'B8 (NIR)', 'B11 (SWIR1)', 'B12 (SWIR2)'],
                    'Reflectance': [
                        spectral_data['B2'],
                        spectral_data['B3'],
                        spectral_data['B4'],
                        spectral_data['B8'],
                        spectral_data['B11'],
                        spectral_data['B12']
                    ]
                })
                st.bar_chart(bands_df.set_index('Band'))
            
            with tab2:
                st.subheader("Geographic Location")
                
                # Create folium map
                m = folium.Map(location=[lat, lon], zoom_start=12)
                
                # Add marker with color based on legal status
                marker_color = 'red' if legal_status['alert_level'] == 'RED' else 'green'
                folium.Marker(
                    [lat, lon],
                    popup=f"{classification['prediction']} - {legal_status['status']}",
                    icon=folium.Icon(color=marker_color, icon='info-sign')
                ).add_to(m)
                
                # Add permitted sites
                for site in PERMITTED_SITES:
                    folium.Marker(
                        [site['lat'], site['lon']],
                        popup=f"✅ {site['name']}<br>Permit: {site['permit_id']}",
                        icon=folium.Icon(color='blue', icon='ok-sign')
                    ).add_to(m)
                
                folium_static(m)
            
            with tab3:
                st.subheader("Predictive Mineral Mapping")
                st.info("💡 Using spectral signatures to predict potential mineral deposits")
                
                for pred in mineral_predictions:
                    with st.expander(f"{pred['mineral']} - {pred['probability']*100:.0f}% Probability"):
                        st.write(f"**Indicator:** {pred['indicator']}")
                        st.progress(pred['probability'])
                        
                        if pred['probability'] > 0.7:
                            st.success("🎯 High-value exploration target")
                        elif pred['probability'] > 0.5:
                            st.warning("⚠️ Moderate potential - further investigation recommended")
                        else:
                            st.info("ℹ️ Low probability - routine monitoring")
            
            with tab4:
                st.subheader("Evidence Package")
                
                if legal_status['alert_level'] == 'RED':
                    st.error("🚨 **VIOLATION DETECTED** - Evidence package generated")
                    
                    # Generate full evidence report
                    evidence = generate_evidence_report(lat, lon, spectral_data, classification, legal_status)
                    
                    # Display evidence in structured format
                    st.json(evidence)
                    
                    # Historical timeline
                    st.subheader("📅 Historical Activity Timeline")
                    timeline_df = pd.DataFrame(evidence['historical_activity'])
                    st.dataframe(timeline_df, use_container_width=True)
                    
                    # Download button for evidence
                    st.download_button(
                        "📥 Download Evidence Report (JSON)",
                        data=json.dumps(evidence, indent=2),
                        file_name=f"evidence_report_{lat}_{lon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.success("✅ No violations detected - Area compliant with regulations")
    
    # Information panel
    with st.sidebar:
        st.divider()
        st.subheader("ℹ️ System Information")
        st.caption("**Data Source:** Sentinel-2 MSI (Real-time when authenticated)")
        st.caption("**GEE Status:** " + ("✅ Authenticated" if gee_available else "⚠️ Using mock data"))
        st.caption("**AI Model:** CNN Transfer Learning (Mock)")
        st.caption("**Target Accuracy:** 92%+")
        st.caption("**Coverage:** Global")
        
        st.divider()
        st.subheader("📊 Permit Database")
        st.caption(f"{len(PERMITTED_SITES)} permitted sites loaded")
        
        for site in PERMITTED_SITES:
            st.caption(f"✓ {site['name']}")

if __name__ == "__main__":
    main()
