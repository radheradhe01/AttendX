# AttendX Landing Page Setup

## 🎨 Font Configuration

The project now uses two Google Fonts:
- **Dosis**: For headings and highlighted elements
- **Manrope**: For body text and descriptions

## 🗄️ MongoDB Setup

### 1. Install MongoDB
```bash
# macOS (using Homebrew)
brew install mongodb-community

# Start MongoDB
brew services start mongodb-community
```

### 2. Environment Variables
Create a `.env.local` file in the root directory:
```env
MONGODB_URI=mongodb://localhost:27017
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🚀 Running the Project

### 1. Install Dependencies
```bash
npm install
```

### 2. Start Development Server
```bash
npm run dev
```

### 3. Access the Application
- **Landing Page**: http://localhost:3000
- **API Documentation**: http://localhost:3000/api/subscribe

## 📁 Project Structure

```
frontend/
├── app/
│   ├── page.tsx              # Landing page
│   ├── layout.tsx             # Root layout with fonts
│   └── api/
│       └── subscribe/
│           └── route.ts       # Email subscription API
├── lib/
│   └── mongodb.ts            # MongoDB connection
├── components/
│   └── ui/                   # shadcn/ui components
└── tailwind.config.js        # Updated with custom fonts
```

## 🎯 Features

### Landing Page
- ✅ Responsive design with Tailwind CSS
- ✅ Hero section with Dosis font
- ✅ Features section with 3 cards
- ✅ Stats section
- ✅ Email subscription form
- ✅ Footer with contact info

### API Features
- ✅ MongoDB integration
- ✅ Email subscription endpoint
- ✅ Duplicate email handling
- ✅ Error handling

### Font Usage
- **Dosis**: Used for headings, titles, and highlighted text
- **Manrope**: Used for body text, descriptions, and general content

## 🔧 Customization

### Adding New Features
1. Create new API routes in `app/api/`
2. Add new components in `components/`
3. Update Tailwind config for new styles

### Database Schema
The `subscribers` collection stores:
```json
{
  "_id": "ObjectId",
  "email": "user@example.com",
  "subscribedAt": "2024-01-01T00:00:00.000Z",
  "source": "landing-page"
}
```

## 🎨 Styling Guidelines

- Use `font-dosis` for headings and important text
- Use `font-manrope` for body text and descriptions
- Follow the existing color scheme (blue, green, purple accents)
- Maintain responsive design principles
