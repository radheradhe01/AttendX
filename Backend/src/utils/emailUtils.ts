import nodemailer from 'nodemailer';
import { cloudConfig } from '../config/cloud';
import { logger } from '../config/logger';
import { CustomError } from '../middleware/errorHandler';

/**
 * Email utility functions
 * Handles email sending for notifications, reports, and system messages
 */

export interface EmailOptions {
  to: string | string[];
  subject: string;
  text?: string;
  html?: string;
  attachments?: Array<{
    filename: string;
    content: Buffer | string;
    contentType?: string;
  }>;
}

export interface EmailTemplate {
  subject: string;
  html: string;
  text: string;
}

/**
 * Email service class
 */
export class EmailService {
  private transporter: nodemailer.Transporter;

  constructor() {
    this.transporter = nodemailer.createTransport({
      host: cloudConfig.email.smtpHost,
      port: cloudConfig.email.smtpPort,
      secure: false, // true for 465, false for other ports
      auth: {
        user: cloudConfig.email.smtpUser,
        pass: cloudConfig.email.smtpPass,
      },
    });
  }

  /**
   * Send email
   */
  async sendEmail(options: EmailOptions): Promise<boolean> {
    try {
      const mailOptions = {
        from: `${cloudConfig.email.fromName} <${cloudConfig.email.fromEmail}>`,
        to: Array.isArray(options.to) ? options.to.join(', ') : options.to,
        subject: options.subject,
        text: options.text,
        html: options.html,
        attachments: options.attachments,
      };

      const result = await this.transporter.sendMail(mailOptions);
      
      logger.info('Email sent successfully:', {
        messageId: result.messageId,
        to: options.to,
        subject: options.subject,
      });

      return true;
    } catch (error) {
      logger.error('Failed to send email:', {
        error: error instanceof Error ? error.message : 'Unknown error',
        to: options.to,
        subject: options.subject,
      });
      
      throw new CustomError('Failed to send email', 500, 'EMAIL_SEND_ERROR');
    }
  }

  /**
   * Send welcome email to new user
   */
  async sendWelcomeEmail(email: string, name: string, role: string): Promise<boolean> {
    const template = this.getWelcomeEmailTemplate(name, role);
    
    return await this.sendEmail({
      to: email,
      subject: template.subject,
      html: template.html,
      text: template.text,
    });
  }

  /**
   * Send password reset email
   */
  async sendPasswordResetEmail(email: string, name: string, resetToken: string): Promise<boolean> {
    const resetUrl = `${process.env.FRONTEND_URL || 'http://localhost:3000'}/reset-password?token=${resetToken}`;
    const template = this.getPasswordResetEmailTemplate(name, resetUrl);
    
    return await this.sendEmail({
      to: email,
      subject: template.subject,
      html: template.html,
      text: template.text,
    });
  }

  /**
   * Send attendance notification email
   */
  async sendAttendanceNotificationEmail(
    email: string,
    studentName: string,
    courseName: string,
    attendanceStatus: string,
    date: string
  ): Promise<boolean> {
    const template = this.getAttendanceNotificationTemplate(
      studentName,
      courseName,
      attendanceStatus,
      date
    );
    
    return await this.sendEmail({
      to: email,
      subject: template.subject,
      html: template.html,
      text: template.text,
    });
  }

  /**
   * Send low attendance warning email
   */
  async sendLowAttendanceWarningEmail(
    email: string,
    studentName: string,
    courseName: string,
    attendancePercentage: number
  ): Promise<boolean> {
    const template = this.getLowAttendanceWarningTemplate(
      studentName,
      courseName,
      attendancePercentage
    );
    
    return await this.sendEmail({
      to: email,
      subject: template.subject,
      html: template.html,
      text: template.text,
    });
  }

  /**
   * Send report ready notification email
   */
  async sendReportReadyEmail(
    email: string,
    reportTitle: string,
    reportType: string,
    downloadUrl: string
  ): Promise<boolean> {
    const template = this.getReportReadyTemplate(reportTitle, reportType, downloadUrl);
    
    return await this.sendEmail({
      to: email,
      subject: template.subject,
      html: template.html,
      text: template.text,
    });
  }

  /**
   * Send system notification email
   */
  async sendSystemNotificationEmail(
    email: string,
    title: string,
    message: string,
    priority: 'low' | 'medium' | 'high' = 'medium'
  ): Promise<boolean> {
    const template = this.getSystemNotificationTemplate(title, message, priority);
    
    return await this.sendEmail({
      to: email,
      subject: template.subject,
      html: template.html,
      text: template.text,
    });
  }

  /**
   * Test email configuration
   */
  async testEmailConfiguration(): Promise<boolean> {
    try {
      await this.transporter.verify();
      logger.info('Email configuration is valid');
      return true;
    } catch (error) {
      logger.error('Email configuration test failed:', error);
      return false;
    }
  }

  /**
   * Get welcome email template
   */
  private getWelcomeEmailTemplate(name: string, role: string): EmailTemplate {
    const subject = `Welcome to AttendEase - ${role.charAt(0).toUpperCase() + role.slice(1)} Account Created`;
    
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>Welcome to AttendEase</title>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background-color: #4CAF50; color: white; padding: 20px; text-align: center; }
          .content { padding: 20px; background-color: #f9f9f9; }
          .footer { padding: 20px; text-align: center; font-size: 12px; color: #666; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>Welcome to AttendEase</h1>
          </div>
          <div class="content">
            <h2>Hello ${name}!</h2>
            <p>Welcome to AttendEase, the Smart College ERP System with Integrated Attendance Management.</p>
            <p>Your ${role} account has been successfully created. You can now access the system using your credentials.</p>
            <p><strong>Account Details:</strong></p>
            <ul>
              <li>Role: ${role.charAt(0).toUpperCase() + role.slice(1)}</li>
              <li>Status: Active</li>
            </ul>
            <p>Please log in to your account and complete your profile setup.</p>
            <p>If you have any questions or need assistance, please contact the system administrator.</p>
          </div>
          <div class="footer">
            <p>This is an automated message from AttendEase System. Please do not reply to this email.</p>
          </div>
        </div>
      </body>
      </html>
    `;

    const text = `
      Welcome to AttendEase!
      
      Hello ${name}!
      
      Welcome to AttendEase, the Smart College ERP System with Integrated Attendance Management.
      
      Your ${role} account has been successfully created. You can now access the system using your credentials.
      
      Account Details:
      - Role: ${role.charAt(0).toUpperCase() + role.slice(1)}
      - Status: Active
      
      Please log in to your account and complete your profile setup.
      
      If you have any questions or need assistance, please contact the system administrator.
      
      This is an automated message from AttendEase System.
    `;

    return { subject, html, text };
  }

  /**
   * Get password reset email template
   */
  private getPasswordResetEmailTemplate(name: string, resetUrl: string): EmailTemplate {
    const subject = 'AttendEase - Password Reset Request';
    
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>Password Reset</title>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background-color: #f44336; color: white; padding: 20px; text-align: center; }
          .content { padding: 20px; background-color: #f9f9f9; }
          .button { display: inline-block; padding: 12px 24px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 4px; margin: 20px 0; }
          .footer { padding: 20px; text-align: center; font-size: 12px; color: #666; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>Password Reset Request</h1>
          </div>
          <div class="content">
            <h2>Hello ${name}!</h2>
            <p>We received a request to reset your password for your AttendEase account.</p>
            <p>Click the button below to reset your password:</p>
            <p><a href="${resetUrl}" class="button">Reset Password</a></p>
            <p>If the button doesn't work, copy and paste this link into your browser:</p>
            <p>${resetUrl}</p>
            <p><strong>Important:</strong> This link will expire in 1 hour for security reasons.</p>
            <p>If you didn't request this password reset, please ignore this email or contact support if you have concerns.</p>
          </div>
          <div class="footer">
            <p>This is an automated message from AttendEase System. Please do not reply to this email.</p>
          </div>
        </div>
      </body>
      </html>
    `;

    const text = `
      Password Reset Request
      
      Hello ${name}!
      
      We received a request to reset your password for your AttendEase account.
      
      To reset your password, please visit the following link:
      ${resetUrl}
      
      Important: This link will expire in 1 hour for security reasons.
      
      If you didn't request this password reset, please ignore this email or contact support if you have concerns.
      
      This is an automated message from AttendEase System.
    `;

    return { subject, html, text };
  }

  /**
   * Get attendance notification email template
   */
  private getAttendanceNotificationTemplate(
    studentName: string,
    courseName: string,
    attendanceStatus: string,
    date: string
  ): EmailTemplate {
    const subject = `AttendEase - Attendance Marked for ${courseName}`;
    
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>Attendance Notification</title>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background-color: #2196F3; color: white; padding: 20px; text-align: center; }
          .content { padding: 20px; background-color: #f9f9f9; }
          .status { padding: 10px; border-radius: 4px; margin: 10px 0; }
          .status.present { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
          .status.late { background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
          .status.absent { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
          .footer { padding: 20px; text-align: center; font-size: 12px; color: #666; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>Attendance Notification</h1>
          </div>
          <div class="content">
            <h2>Hello ${studentName}!</h2>
            <p>Your attendance has been marked for the following class:</p>
            <p><strong>Course:</strong> ${courseName}</p>
            <p><strong>Date:</strong> ${date}</p>
            <div class="status ${attendanceStatus.toLowerCase()}">
              <strong>Status:</strong> ${attendanceStatus.toUpperCase()}
            </div>
            <p>You can view your complete attendance record by logging into your AttendEase account.</p>
          </div>
          <div class="footer">
            <p>This is an automated message from AttendEase System. Please do not reply to this email.</p>
          </div>
        </div>
      </body>
      </html>
    `;

    const text = `
      Attendance Notification
      
      Hello ${studentName}!
      
      Your attendance has been marked for the following class:
      
      Course: ${courseName}
      Date: ${date}
      Status: ${attendanceStatus.toUpperCase()}
      
      You can view your complete attendance record by logging into your AttendEase account.
      
      This is an automated message from AttendEase System.
    `;

    return { subject, html, text };
  }

  /**
   * Get low attendance warning email template
   */
  private getLowAttendanceWarningTemplate(
    studentName: string,
    courseName: string,
    attendancePercentage: number
  ): EmailTemplate {
    const subject = `AttendEase - Low Attendance Warning for ${courseName}`;
    
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>Low Attendance Warning</title>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background-color: #ff9800; color: white; padding: 20px; text-align: center; }
          .content { padding: 20px; background-color: #f9f9f9; }
          .warning { background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; padding: 15px; border-radius: 4px; margin: 15px 0; }
          .footer { padding: 20px; text-align: center; font-size: 12px; color: #666; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>Low Attendance Warning</h1>
          </div>
          <div class="content">
            <h2>Hello ${studentName}!</h2>
            <div class="warning">
              <h3>⚠️ Attendance Warning</h3>
              <p>Your attendance percentage for <strong>${courseName}</strong> is currently <strong>${attendancePercentage.toFixed(1)}%</strong>, which is below the minimum required attendance.</p>
            </div>
            <p>Please ensure you attend all future classes to maintain good attendance. Regular attendance is important for your academic success.</p>
            <p>You can view your detailed attendance record by logging into your AttendEase account.</p>
            <p>If you have any concerns or need to discuss your attendance, please contact your faculty or academic advisor.</p>
          </div>
          <div class="footer">
            <p>This is an automated message from AttendEase System. Please do not reply to this email.</p>
          </div>
        </div>
      </body>
      </html>
    `;

    const text = `
      Low Attendance Warning
      
      Hello ${studentName}!
      
      ⚠️ ATTENDANCE WARNING
      
      Your attendance percentage for ${courseName} is currently ${attendancePercentage.toFixed(1)}%, which is below the minimum required attendance.
      
      Please ensure you attend all future classes to maintain good attendance. Regular attendance is important for your academic success.
      
      You can view your detailed attendance record by logging into your AttendEase account.
      
      If you have any concerns or need to discuss your attendance, please contact your faculty or academic advisor.
      
      This is an automated message from AttendEase System.
    `;

    return { subject, html, text };
  }

  /**
   * Get report ready email template
   */
  private getReportReadyTemplate(
    reportTitle: string,
    reportType: string,
    downloadUrl: string
  ): EmailTemplate {
    const subject = `AttendEase - Report Ready: ${reportTitle}`;
    
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>Report Ready</title>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background-color: #4CAF50; color: white; padding: 20px; text-align: center; }
          .content { padding: 20px; background-color: #f9f9f9; }
          .button { display: inline-block; padding: 12px 24px; background-color: #2196F3; color: white; text-decoration: none; border-radius: 4px; margin: 20px 0; }
          .footer { padding: 20px; text-align: center; font-size: 12px; color: #666; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>Report Ready</h1>
          </div>
          <div class="content">
            <h2>Your report is ready!</h2>
            <p>The report you requested has been generated successfully.</p>
            <p><strong>Report Details:</strong></p>
            <ul>
              <li>Title: ${reportTitle}</li>
              <li>Type: ${reportType.charAt(0).toUpperCase() + reportType.slice(1)}</li>
              <li>Generated: ${new Date().toLocaleDateString()}</li>
            </ul>
            <p>Click the button below to download your report:</p>
            <p><a href="${downloadUrl}" class="button">Download Report</a></p>
            <p><strong>Note:</strong> This download link will expire in 30 days for security reasons.</p>
          </div>
          <div class="footer">
            <p>This is an automated message from AttendEase System. Please do not reply to this email.</p>
          </div>
        </div>
      </body>
      </html>
    `;

    const text = `
      Report Ready
      
      Your report is ready!
      
      The report you requested has been generated successfully.
      
      Report Details:
      - Title: ${reportTitle}
      - Type: ${reportType.charAt(0).toUpperCase() + reportType.slice(1)}
      - Generated: ${new Date().toLocaleDateString()}
      
      Download your report: ${downloadUrl}
      
      Note: This download link will expire in 30 days for security reasons.
      
      This is an automated message from AttendEase System.
    `;

    return { subject, html, text };
  }

  /**
   * Get system notification email template
   */
  private getSystemNotificationTemplate(
    title: string,
    message: string,
    priority: 'low' | 'medium' | 'high'
  ): EmailTemplate {
    const priorityColors = {
      low: '#4CAF50',
      medium: '#ff9800',
      high: '#f44336'
    };

    const subject = `AttendEase - ${title}`;
    
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>System Notification</title>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background-color: ${priorityColors[priority]}; color: white; padding: 20px; text-align: center; }
          .content { padding: 20px; background-color: #f9f9f9; }
          .footer { padding: 20px; text-align: center; font-size: 12px; color: #666; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>System Notification</h1>
            <p>Priority: ${priority.toUpperCase()}</p>
          </div>
          <div class="content">
            <h2>${title}</h2>
            <p>${message}</p>
          </div>
          <div class="footer">
            <p>This is an automated message from AttendEase System. Please do not reply to this email.</p>
          </div>
        </div>
      </body>
      </html>
    `;

    const text = `
      System Notification
      Priority: ${priority.toUpperCase()}
      
      ${title}
      
      ${message}
      
      This is an automated message from AttendEase System.
    `;

    return { subject, html, text };
  }
}

// Export singleton instance
export const emailService = new EmailService();
