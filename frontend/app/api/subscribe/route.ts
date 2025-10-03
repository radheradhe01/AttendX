import { NextRequest, NextResponse } from 'next/server'
import { connectToDatabase } from '@/lib/mongodb'

export async function POST(request: NextRequest) {
  try {
    const { email } = await request.json()

    if (!email || !email.includes('@')) {
      return NextResponse.json(
        { error: 'Valid email is required' },
        { status: 400 }
      )
    }

    const { db } = await connectToDatabase()
    const subscribers = db.collection('subscribers')

    // Check if email already exists
    const existingSubscriber = await subscribers.findOne({ email })
    if (existingSubscriber) {
      return NextResponse.json(
        { message: 'Email already subscribed' },
        { status: 200 }
      )
    }

    // Insert new subscriber
    const result = await subscribers.insertOne({
      email,
      subscribedAt: new Date(),
      source: 'landing-page'
    })

    return NextResponse.json(
      { 
        message: 'Successfully subscribed!',
        id: result.insertedId 
      },
      { status: 201 }
    )

  } catch (error) {
    console.error('Subscription error:', error)
    return NextResponse.json(
      { error: 'Failed to subscribe. Please try again.' },
      { status: 500 }
    )
  }
}
