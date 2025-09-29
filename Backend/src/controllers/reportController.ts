import { Request, Response, NextFunction } from 'express';
import { Report } from '../models/Report';
import { erpService } from '../services/erpService';
import { logger } from '../config/logger';
import { CustomError, asyncHandler } from '../middleware/errorHandler';
import { validationResult } from 'express-validator';
import fs from 'fs';
import path from 'path';
import { Parser } from 'json2csv';
import PDFDocument from 'pdfkit';

/**
 * Report Controller
 * Handles report generation, management, and download
 */

export interface GenerateReportRequest extends Request {
  body: {
    reportType: 'attendance' | 'academic' | 'financial' | 'student' | 'faculty' | 'course';
    title: string;
    description: string;
    parameters: {
      startDate?: string;
      endDate?: string;
      department?: string;
      course?: string;
      student?: string;
      faculty?: string;
      semester?: number;
      year?: number;
      format: 'pdf' | 'csv' | 'excel';
    };
    isPublic?: boolean;
    tags?: string[];
  };
}

/**
 * @desc    Generate a new report
 * @route   POST /api/reports/generate
 * @access  Private (Faculty, Admin)
 */
export const generateReport = asyncHandler(async (req: GenerateReportRequest, res: Response, next: NextFunction) => {
  // Check for validation errors
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      success: false,
      message: 'Validation failed',
      errors: errors.array(),
      code: 'VALIDATION_ERROR'
    });
  }

  const { reportType, title, description, parameters, isPublic = false, tags = [] } = req.body;
  const generatedBy = req.user?.id;

  logger.info('Generating report:', { reportType, title, generatedBy });

  try {
    // Create report record
    const report = await Report.create({
      reportType,
      title,
      description,
      generatedBy,
      parameters: {
        ...parameters,
        startDate: parameters.startDate ? new Date(parameters.startDate) : undefined,
        endDate: parameters.endDate ? new Date(parameters.endDate) : undefined,
      },
      fileInfo: {
        fileName: '',
        filePath: '',
        fileSize: 0,
        mimeType: '',
        downloadCount: 0,
      },
      status: 'generating',
      isPublic,
      tags,
    });

    // Generate report data based on type
    let reportData: any[] = [];
    let fileName = '';

    switch (reportType) {
      case 'attendance':
        reportData = await erpService.generateAttendanceReport(
          parameters.startDate ? new Date(parameters.startDate) : new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // Default to last 30 days
          parameters.endDate ? new Date(parameters.endDate) : new Date(),
          parameters.department,
          parameters.course
        );
        fileName = `attendance_report_${Date.now()}`;
        break;

      case 'academic':
        reportData = await erpService.getAcademicPerformance(
          parameters.department,
          parameters.semester,
          parameters.year
        );
        fileName = `academic_performance_${Date.now()}`;
        break;

      case 'financial':
        const feeStats = await erpService.getFeeCollectionStats();
        reportData = [feeStats];
        fileName = `financial_report_${Date.now()}`;
        break;

      case 'student':
        // This would typically fetch student data based on parameters
        reportData = [];
        fileName = `student_report_${Date.now()}`;
        break;

      case 'faculty':
        // This would typically fetch faculty data based on parameters
        reportData = [];
        fileName = `faculty_report_${Date.now()}`;
        break;

      case 'course':
        // This would typically fetch course data based on parameters
        reportData = [];
        fileName = `course_report_${Date.now()}`;
        break;

      default:
        throw new CustomError('Invalid report type', 400, 'INVALID_REPORT_TYPE');
    }

    // Generate file based on format
    const { filePath, fileSize, mimeType } = await generateReportFile(
      reportData,
      fileName,
      parameters.format,
      reportType
    );

    // Update report with file information
    const finalFileName = `${fileName}.${parameters.format}`;
    await Report.findByIdAndUpdate(report._id, {
      'fileInfo.fileName': finalFileName,
      'fileInfo.filePath': filePath,
      'fileInfo.fileSize': fileSize,
      'fileInfo.mimeType': mimeType,
      status: 'completed',
    });

    logger.info('Report generated successfully:', {
      reportId: report._id,
      reportType,
      fileName: finalFileName,
    });

    return res.status(201).json({
      success: true,
      message: 'Report generated successfully',
      data: {
        report: {
          id: report._id,
          reportType,
          title,
          status: 'completed',
          fileName: finalFileName,
          fileSize,
          downloadUrl: `/api/reports/${report._id}/download`,
        },
      },
    });

  } catch (error) {
    logger.error('Report generation failed:', error);

    // Update report status to failed
    if (req.body.reportType) {
      await Report.findOneAndUpdate(
        { generatedBy, reportType: req.body.reportType, status: 'generating' },
        {
          status: 'failed',
          errorMessage: error instanceof Error ? error.message : 'Unknown error',
        }
      );
    }

    throw error;
  }
});

/**
 * @desc    Get all reports
 * @route   GET /api/reports
 * @access  Private (Faculty, Admin)
 */
export const getReports = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const {
    page = 1,
    limit = 10,
    reportType,
    status,
    generatedBy,
    search
  } = req.query;

  const query: any = {};

  // Apply filters
  if (reportType) {
    query.reportType = reportType;
  }
  if (status) {
    query.status = status;
  }
  if (generatedBy) {
    query.generatedBy = generatedBy;
  }
  if (search) {
    query.$or = [
      { title: { $regex: search, $options: 'i' } },
      { description: { $regex: search, $options: 'i' } },
    ];
  }

  // If user is not admin, only show their own reports and public reports
  if (req.user?.role !== 'admin') {
    query.$or = [
      { generatedBy: req.user?.id },
      { isPublic: true }
    ];
  }

  const skip = (Number(page) - 1) * Number(limit);

  const [reports, total] = await Promise.all([
    Report.find(query)
      .populate('generatedBy', 'email role')
      .select('-__v')
      .sort({ generatedAt: -1 })
      .skip(skip)
      .limit(Number(limit)),
    Report.countDocuments(query)
  ]);

  return res.json({
    success: true,
    data: {
      reports,
      pagination: {
        current: Number(page),
        pages: Math.ceil(total / Number(limit)),
        total,
      },
    },
  });
});

/**
 * @desc    Get report by ID
 * @route   GET /api/reports/:id
 * @access  Private (Faculty, Admin)
 */
export const getReportById = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;

  const report = await Report.findById(id)
    .populate('generatedBy', 'email role');

  if (!report) {
    return res.status(400).json({ success: false, message: 'Report not found', code: 'REPORT_NOT_FOUND' });
  }

  // Check access permissions
  if (req.user?.role !== 'admin' && 
      report.generatedBy._id.toString() !== req.user?.id && 
      !report.isPublic) {
    return res.status(400).json({
      success: false,
      message: 'Access denied to this report',
      code: 'ACCESS_DENIED'
    });
  }

  return res.json({
    success: true,
    data: {
      report,
    },
  });
});

/**
 * @desc    Download report file
 * @route   GET /api/reports/:id/download
 * @access  Private (Faculty, Admin)
 */
export const downloadReport = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;

  const report = await Report.findById(id);

  if (!report) {
    return res.status(400).json({ success: false, message: 'Report not found', code: 'REPORT_NOT_FOUND' });
  }

  // Check access permissions
  if (req.user?.role !== 'admin' && 
      report.generatedBy.toString() !== req.user?.id && 
      !report.isPublic) {
    return res.status(400).json({
      success: false,
      message: 'Access denied to this report',
      code: 'ACCESS_DENIED'
    });
  }

  // Check if report is completed
  if (report.status !== 'completed') {
    return res.status(400).json({
      success: false,
      message: 'Report is not ready for download',
      code: 'REPORT_NOT_READY'
    });
  }

  // Check if file exists
  if (!fs.existsSync(report.fileInfo.filePath)) {
    return res.status(400).json({ success: false, message: 'Report file not found', code: 'FILE_NOT_FOUND' });
  }

  // Increment download count
  await Report.findByIdAndUpdate(id, {
    $inc: { 'fileInfo.downloadCount': 1 }
  });

  // Set headers for file download
  res.setHeader('Content-Type', report.fileInfo.mimeType);
  res.setHeader('Content-Disposition', `attachment; filename="${report.fileInfo.fileName}"`);
  res.setHeader('Content-Length', report.fileInfo.fileSize);

  // Stream the file
  const fileStream = fs.createReadStream(report.fileInfo.filePath);
  fileStream.pipe(res);

  logger.info('Report downloaded:', {
    reportId: report._id,
    fileName: report.fileInfo.fileName,
    downloadedBy: req.user?.id
  });
  
  return;
});

/**
 * @desc    Delete report
 * @route   DELETE /api/reports/:id
 * @access  Private (Faculty, Admin)
 */
export const deleteReport = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { id } = req.params;

  const report = await Report.findById(id);

  if (!report) {
    return res.status(400).json({ success: false, message: 'Report not found', code: 'REPORT_NOT_FOUND' });
  }

  // Check access permissions - only admin or report owner can delete
  if (req.user?.role !== 'admin' && report.generatedBy.toString() !== req.user?.id) {
    return res.status(400).json({
      success: false,
      message: 'Access denied to delete this report',
      code: 'ACCESS_DENIED'
    });
  }

  // Delete file if it exists
  if (fs.existsSync(report.fileInfo.filePath)) {
    fs.unlinkSync(report.fileInfo.filePath);
  }

  // Delete report record
  await Report.findByIdAndDelete(id);

  logger.info('Report deleted:', {
    reportId: report._id,
    fileName: report.fileInfo.fileName,
    deletedBy: req.user?.id
  });

  return res.json({
    success: true,
    message: 'Report deleted successfully',
  });
});

/**
 * @desc    Get dashboard statistics
 * @route   GET /api/reports/dashboard
 * @access  Private (Admin)
 */
export const getDashboardStats = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const stats = await erpService.getDashboardStats();

  return res.json({
    success: true,
    data: {
      stats,
    },
  });
});

/**
 * @desc    Get low attendance students
 * @route   GET /api/reports/low-attendance
 * @access  Private (Faculty, Admin)
 */
export const getLowAttendanceStudents = asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
  const { threshold = 75 } = req.query;

  const lowAttendanceStudents = await erpService.getLowAttendanceStudents(Number(threshold));

  return res.json({
    success: true,
    data: {
      students: lowAttendanceStudents,
      threshold: Number(threshold),
      count: lowAttendanceStudents.length,
    },
  });
});

/**
 * Helper function to generate report file
 */
async function generateReportFile(
  data: any[],
  fileName: string,
  format: string,
  reportType: string
): Promise<{ filePath: string; fileSize: number; mimeType: string }> {
  const reportsDir = path.join(process.cwd(), 'uploads', 'reports');
  
  // Ensure reports directory exists
  if (!fs.existsSync(reportsDir)) {
    fs.mkdirSync(reportsDir, { recursive: true });
  }

  let filePath: string;
  let mimeType: string;

  switch (format) {
    case 'csv':
      filePath = path.join(reportsDir, `${fileName}.csv`);
      mimeType = 'text/csv';
      
      if (data.length > 0) {
        const fields = Object.keys(data[0]);
        const parser = new Parser({ fields });
        const csv = parser.parse(data);
        fs.writeFileSync(filePath, csv);
      } else {
        fs.writeFileSync(filePath, '');
      }
      break;

    case 'pdf':
      filePath = path.join(reportsDir, `${fileName}.pdf`);
      mimeType = 'application/pdf';
      
      const doc = new PDFDocument();
      doc.pipe(fs.createWriteStream(filePath));
      
      // Add title
      doc.fontSize(20).text(`${reportType.toUpperCase()} REPORT`, 50, 50);
      doc.fontSize(12).text(`Generated on: ${new Date().toLocaleDateString()}`, 50, 80);
      doc.moveDown();
      
      // Add data
      if (data.length > 0) {
        data.forEach((item, index) => {
          doc.text(`${index + 1}. ${JSON.stringify(item, null, 2)}`);
          doc.moveDown();
        });
      } else {
        doc.text('No data available');
      }
      
      doc.end();
      break;

    case 'excel':
      // For Excel, we'll create a CSV file with .xlsx extension
      // In a real implementation, you'd use a library like 'xlsx'
      filePath = path.join(reportsDir, `${fileName}.xlsx`);
      mimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
      
      if (data.length > 0) {
        const fields = Object.keys(data[0]);
        const parser = new Parser({ fields });
        const csv = parser.parse(data);
        fs.writeFileSync(filePath, csv);
      } else {
        fs.writeFileSync(filePath, '');
      }
      break;

    default:
      throw new CustomError('Unsupported file format', 400, 'UNSUPPORTED_FORMAT');
  }

  const fileSize = fs.statSync(filePath).size;

  return { filePath, fileSize, mimeType };
}
