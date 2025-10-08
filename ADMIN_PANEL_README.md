# 🔐 Admin Panel Documentation

## Overview
The Restaurant Finder Admin Panel provides comprehensive analytics and user interaction tracking for the restaurant recommendation platform.

## Access
- **URL**: `/admin`
- **Username**: `admin`
- **Password**: `admin`

## Features

### 🔐 Secure Authentication
- Session-based authentication
- Protected routes with login requirement
- Secure logout functionality

### 📊 Analytics Dashboard
- **Total Interactions**: Count of all user interactions
- **Unique Users**: Number of unique IP addresses
- **Search Analytics**: Most searched items and queries
- **Category Analytics**: Most clicked categories
- **Chatbot Analytics**: Chatbot usage and popular queries
- **Recent Activity**: Real-time activity feed

### 📈 Data Visualization
- **Statistics Cards**: Key metrics at a glance
- **Data Tables**: Detailed breakdowns of user behavior
- **Usage Charts**: Visual representation of peak usage times
- **Export Functionality**: Download analytics data as JSON

### 🎯 User Interaction Tracking
The admin panel automatically tracks:
- **Search Queries**: All user search inputs
- **Category Clicks**: Category-based filtering interactions
- **Chatbot Queries**: Conversational AI interactions
- **Timestamps**: When interactions occurred
- **IP Addresses**: User identification (anonymized)
- **User Agents**: Browser and device information

## Technical Implementation

### Backend Routes
```python
@app.route('/admin')                    # Login page
@app.route('/admin/login', methods=['POST'])  # Authentication
@app.route('/admin/dashboard')          # Analytics dashboard
@app.route('/admin/logout')             # Logout
```

### Data Storage
- **File**: `data/user_interactions.json`
- **Format**: JSON with interaction logs
- **Retention**: Last 1000 interactions (auto-cleanup)
- **Structure**: Timestamp, type, data, IP, user agent

### Security Features
- **Session Management**: Flask session-based authentication
- **Access Control**: Route protection with login checks
- **Data Privacy**: IP address tracking for analytics only
- **Auto-cleanup**: Prevents data file from growing too large

## Usage Guide

### 1. Accessing the Admin Panel
1. Navigate to `/admin` in your browser
2. Enter username: `admin`
3. Enter password: `admin`
4. Click "Login to Dashboard"

### 2. Viewing Analytics
- **Overview**: Check the statistics cards for key metrics
- **Search Data**: View most popular search queries
- **Category Data**: See which categories are most clicked
- **Chatbot Data**: Analyze conversational AI usage
- **Recent Activity**: Monitor real-time user interactions

### 3. Data Export
- Click "Export Data" button to download analytics
- Data is exported as JSON format
- Includes all tracked interactions and metrics

### 4. Auto-Refresh
- Dashboard automatically refreshes every 5 minutes
- Manual refresh available via "Refresh Data" button

## Data Privacy

### What's Tracked
- Search queries (for improving recommendations)
- Category clicks (for understanding user preferences)
- Chatbot interactions (for improving AI responses)
- Timestamps (for usage pattern analysis)
- IP addresses (for unique user counting)

### What's NOT Tracked
- Personal information
- User accounts or profiles
- Sensitive data
- Payment information

### Data Retention
- Interactions are kept for the last 1000 entries
- Older data is automatically removed
- No permanent storage of user data

## Customization

### Adding New Tracking
To track new user interactions, add this to your Flask route:
```python
log_user_interaction('interaction_type', {'key': 'value'})
```

### Modifying Analytics
Edit the `admin_dashboard()` function in `app.py` to:
- Add new metrics
- Change data visualization
- Modify data retention policies
- Add new export formats

### Styling Changes
- Edit `admin_dashboard.html` for layout changes
- Modify CSS in the `<style>` section
- Add new JavaScript functionality as needed

## Security Considerations

### Production Deployment
- Change default admin credentials
- Use environment variables for sensitive data
- Implement proper session management
- Add HTTPS for secure communication
- Consider IP whitelisting for admin access

### Access Control
- Monitor admin panel access logs
- Implement role-based permissions if needed
- Add two-factor authentication for enhanced security
- Regular security audits

## Troubleshooting

### Common Issues
1. **Login not working**: Check username/password are exactly `admin`/`admin`
2. **No data showing**: Ensure user interactions are being tracked
3. **Dashboard not loading**: Check Flask session configuration
4. **Export not working**: Verify browser allows file downloads

### Debug Mode
- Check Flask console for error messages
- Verify `data/user_interactions.json` file exists
- Ensure proper file permissions for data directory

## API Endpoints

### Admin Routes
- `GET /admin` - Login page
- `POST /admin/login` - Authentication
- `GET /admin/dashboard` - Analytics dashboard
- `GET /admin/logout` - Logout

### Data Endpoints
- User interactions are automatically logged
- No additional API endpoints required
- Data is stored in JSON format

## Future Enhancements

### Planned Features
- Real-time analytics with WebSocket
- Advanced data visualization with charts
- User behavior heatmaps
- A/B testing analytics
- Performance metrics tracking

### Integration Options
- Google Analytics integration
- Database storage for larger datasets
- Machine learning insights
- Automated reporting
- Email notifications for key metrics

---

*This admin panel provides essential analytics for understanding user behavior and improving the Restaurant Finder platform.*
