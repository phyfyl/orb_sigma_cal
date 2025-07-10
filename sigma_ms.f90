MODULE kinds
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(P=15, R=307) ! Double precision
END MODULE kinds

! mpi_module_interface.f90
MODULE mpi
  USE kinds
  IMPLICIT NONE
  INCLUDE 'mpif.h' ! Standard MPI include file

  INTEGER :: my_rank
  INTEGER :: num_procs
  INTEGER :: ierr_mpi

CONTAINS
  SUBROUTINE mpi_setup()
    CALL MPI_Init(ierr_mpi)
    CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr_mpi)
    CALL MPI_Comm_size(MPI_COMM_WORLD, num_procs, ierr_mpi)
  END SUBROUTINE mpi_setup

  SUBROUTINE mpi_finalize_custom()
    CALL MPI_Finalize(ierr_mpi)
  END SUBROUTINE mpi_finalize_custom

  SUBROUTINE mpi_reduce_sum_cmplx(send_buf, recv_buf, count)
    COMPLEX(dp), INTENT(IN) :: send_buf(*) ! Assumed-size array
    COMPLEX(dp), INTENT(OUT), OPTIONAL :: recv_buf(*) ! Assumed-size, only relevant on root
    INTEGER, INTENT(IN) :: count

    IF (my_rank == 0) THEN
        IF (.NOT. PRESENT(recv_buf)) THEN
            PRINT *, "Error: recv_buf not present on root for MPI_Reduce"
            CALL MPI_Abort(MPI_COMM_WORLD, 99, ierr_mpi)
        END IF
        CALL MPI_Reduce(send_buf, recv_buf, count, MPI_DOUBLE_COMPLEX, MPI_SUM,0, MPI_COMM_WORLD, ierr_mpi)
    ELSE
        CALL MPI_Reduce(send_buf, send_buf, count, MPI_DOUBLE_COMPLEX, MPI_SUM,0, MPI_COMM_WORLD, ierr_mpi)
    END IF
  END SUBROUTINE mpi_reduce_sum_cmplx

END MODULE mpi


MODULE constants
  USE kinds
  IMPLICIT NONE

  COMPLEX(dp), PARAMETER :: I_UNIT = CMPLX(0.0_dp, 1.0_dp, KIND=dp)

  ! Pauli matrices (as complex type for consistency)
  COMPLEX(dp), PARAMETER :: s0(2,2) = RESHAPE([CMPLX(1.0_dp,0.0_dp,dp), CMPLX(0.0_dp,0.0_dp,dp), &
                                                CMPLX(0.0_dp,0.0_dp,dp), CMPLX(1.0_dp,0.0_dp,dp)], [2,2])
  COMPLEX(dp), PARAMETER :: sx(2,2) = RESHAPE([CMPLX(0.0_dp,0.0_dp,dp), CMPLX(1.0_dp,0.0_dp,dp), &
                                                CMPLX(1.0_dp,0.0_dp,dp), CMPLX(0.0_dp,0.0_dp,dp)], [2,2])
  COMPLEX(dp), PARAMETER :: sy(2,2) = RESHAPE([CMPLX(0.0_dp,0.0_dp,dp), CMPLX(0.0_dp,1.0_dp,dp), &
                                                CMPLX(0.0_dp,-1.0_dp,dp),CMPLX(0.0_dp,0.0_dp,dp)], [2,2])
  COMPLEX(dp), PARAMETER :: sz(2,2) = RESHAPE([CMPLX(1.0_dp,0.0_dp,dp), CMPLX(0.0_dp,0.0_dp,dp), &
                                                CMPLX(0.0_dp,0.0_dp,dp), CMPLX(-1.0_dp,0.0_dp,dp)], [2,2])

  ! Physical parameters
  REAL(dp), PARAMETER :: v  = 0.76_dp
  REAL(dp), PARAMETER :: Js = 0.036_dp
  REAL(dp), PARAMETER :: Jd = 0.029_dp
  REAL(dp), PARAMETER :: Ds = 0.084_dp
  REAL(dp), PARAMETER :: Dd = -0.127_dp
  REAL(dp), PARAMETER :: m0 = Js ! Renamed from m0_param to avoid conflict
  REAL(dp), PARAMETER :: m1 = Js - Jd ! Renamed from m1_param
  REAL(dp), PARAMETER :: dsl = 10.8377066667_dp
  REAL(dp), PARAMETER :: dsp = 2.79896_dp
  ! epsilon_r not used

  REAL(dp), PARAMETER :: echarge = 1.602176634E-19_dp
  REAL(dp), PARAMETER :: h = 6.62607015E-34_dp
  ! kB not used
  REAL(dp), PARAMETER :: e2htoS = echarge**2 / h
  REAL(dp), PARAMETER :: Pi = ACOS(-1.0_dp)

CONTAINS
  SUBROUTINE kronecker_product_cmplx_cmplx(A, B, C, m1, n1, m2, n2)
    COMPLEX(dp), INTENT(IN) :: A(m1,n1), B(m2,n2)
    COMPLEX(dp), INTENT(OUT) :: C(m1*m2, n1*n2)
    INTEGER, INTENT(IN) :: m1, n1, m2, n2
    INTEGER :: i, j, k, l
    C = CMPLX(0.0_dp, 0.0_dp, dp)
    DO j = 1, n1 ! Column index for A
      DO i = 1, m1 ! Row index for A
        DO l = 1, n2 ! Column index for B
          DO k = 1, m2 ! Row index for B
            C((i-1)*m2 + k, (j-1)*n2 + l) = A(i,j) * B(k,l)
          END DO
        END DO
      END DO
    END DO
  END SUBROUTINE kronecker_product_cmplx_cmplx

  SUBROUTINE kronecker_product_real_cmplx(A_real, B_cmplx, C_cmplx, m1, n1, m2, n2)
    REAL(dp), INTENT(IN) :: A_real(m1,n1)
    COMPLEX(dp), INTENT(IN) :: B_cmplx(m2,n2)
    COMPLEX(dp), INTENT(OUT) :: C_cmplx(m1*m2, n1*n2)
    INTEGER, INTENT(IN) :: m1, n1, m2, n2
    INTEGER :: i, j, k, l
    COMPLEX(dp) :: val_A_cmplx
    C_cmplx = CMPLX(0.0_dp, 0.0_dp, dp)
    DO j = 1, n1
      DO i = 1, m1
        val_A_cmplx = CMPLX(A_real(i,j), 0.0_dp, dp)
        DO l = 1, n2
          DO k = 1, m2
            C_cmplx((i-1)*m2 + k, (j-1)*n2 + l) = val_A_cmplx * B_cmplx(k,l)
          END DO
        END DO
      END DO
    END DO
  END SUBROUTINE kronecker_product_real_cmplx

  SUBROUTINE kronecker_product_real_real(A_real, B_real, C_real, m1, n1, m2, n2)
    REAL(dp), INTENT(IN) :: A_real(m1,n1), B_real(m2,n2)
    REAL(dp), INTENT(OUT) :: C_real(m1*m2, n1*n2)
    INTEGER, INTENT(IN) :: m1, n1, m2, n2
    INTEGER :: i, j, k, l
    C_real = 0.0_dp
    DO j = 1, n1
      DO i = 1, m1
        DO l = 1, n2
          DO k = 1, m2
            C_real((i-1)*m2 + k, (j-1)*n2 + l) = A_real(i,j) * B_real(k,l)
          END DO
        END DO
      END DO
    END DO
  END SUBROUTINE kronecker_product_real_real

END MODULE constants


! physics_subroutines_module.f90
MODULE physics_subroutines
  USE kinds
  USE constants
  IMPLICIT NONE
CONTAINS

  SUBROUTINE calculate_optical_conductivity(Nl,Nbs,us,Bzs,ms,omega, eta, kf, dk_param)
    USE mpi
    INTEGER, INTENT(IN) :: Nl,Nbs
    REAL(dp), INTENT(IN) :: us(2*Nl),ms(Nbs,Nl,3),Bzs(Nbs)
    REAL(dp), INTENT(IN) :: omega, eta, kf, dk_param

    INTEGER :: Nk, M_dim
    REAL(dp), ALLOCATABLE :: kxs(:), kys(:)
    COMPLEX(dp), ALLOCATABLE :: sigma_accum(:, :, :), sigma_s_partial(:, :, :), sigma_a_partial(:, :, :)
    COMPLEX(dp), ALLOCATABLE :: sigma_s_global(:, :, :), sigma_a_global(:, :, :)
    COMPLEX(dp) :: H_k_mat(4*Nl, 4*Nl)
    REAL(dp), ALLOCATABLE :: E_vals(:)
    COMPLEX(dp), ALLOCATABLE :: U_eigvecs_mat(:,:)
    COMPLEX(dp) :: vx_op(4*Nl, 4*Nl), vy_op(4*Nl, 4*Nl)
    COMPLEX(dp), ALLOCATABLE :: vx_eig(:,:), vy_eig(:,:)
    REAL(dp) :: E_n, E_m, fn, fm, delta_E_real
    COMPLEX(dp) :: denominator_c, term_c, prefactor_coeff
    INTEGER :: ikx, iky,ik,ib, n_band, m_band, alpha, beta
    INTEGER :: info, lwork
    COMPLEX(dp), ALLOCATABLE :: work(:)
    REAL(dp), ALLOCATABLE :: rwork(:)
    CHARACTER(LEN=1) :: jobz, uplo
    REAL(dp) :: kx_val, ky_val
    COMPLEX(dp), ALLOCATABLE :: v_alpha_comp(:,:), v_beta_comp(:,:)


    M_dim = 4*Nl
    Nk = INT(2.0_dp * kf / dk_param)
    IF (Nk < 1) Nk = 1 ! Ensure Nk is at least 1

    ALLOCATE(kxs(Nk), kys(Nk))
    ALLOCATE(sigma_accum(Nbs, 2, 2))
    ALLOCATE(sigma_s_partial(Nbs, 2, 2), sigma_a_partial(Nbs, 2, 2))
    IF (my_rank == 0) THEN
      ALLOCATE(sigma_s_global(Nbs, 2, 2), sigma_a_global(Nbs, 2, 2))
    END IF

    ALLOCATE(E_vals(M_dim), U_eigvecs_mat(M_dim, M_dim))
    ALLOCATE(vx_eig(M_dim, M_dim), vy_eig(M_dim, M_dim))
    ALLOCATE(v_alpha_comp(M_dim, M_dim), v_beta_comp(M_dim,M_dim))


    ! LAPACK workspace query and allocation
    jobz = 'V'  ! Compute eigenvalues and eigenvectors
    uplo = 'U'  ! Upper triangle of H_k is stored
    lwork = -1  ! Query optimal size
    ALLOCATE(work(1)) 
    ALLOCATE(rwork(MAX(1, 3*M_dim-2)))
    CALL ZHEEV(jobz, uplo, M_dim, H_k_mat, M_dim, E_vals, work, lwork, rwork, info)
    lwork = INT(work(1)%re)
    DEALLOCATE(work)
    ALLOCATE(work(lwork))

    DO ikx = 1, Nk
      IF (Nk == 1) THEN
        kxs(ikx) = 0.0_dp ! Or -kf if that's the intent for Nk=1
      ELSE
        kxs(ikx) = -kf + REAL(ikx-1, dp) * (2.0_dp * kf) / REAL(Nk-1, dp)
      END IF
    END DO
    kys = kxs

    sigma_accum = CMPLX(0.0_dp, 0.0_dp, dp)

    DO ik=my_rank+1,Nk**2,num_procs
       ikx = (ik-1)/Nk + 1
       iky = ik - (ikx-1)*Nk
       kx_val = kxs(ikx)
       ky_val = kys(iky)
       
       Do ib=1,Nbs
          CALL multi_dirac_cone(Nl, us, ms(ib,:,:),kx_val, ky_val, H_k_mat)
          U_eigvecs_mat = H_k_mat ! ZHEEV needs input in A, output eigenvectors in A
          CALL ZHEEV(jobz, uplo, M_dim, U_eigvecs_mat, M_dim, E_vals, work, lwork, rwork, info)
          IF (info /= 0) THEN
             IF(my_rank == 0) PRINT *, "ZHEEV error, info = ", info, " at kx,ky=", kx_val, ky_val
             CYCLE ! Or stop
          ENDIF

          CALL dh_dk(Nl, vx_op, vy_op)
          vx_eig = MATMUL(CONJG(TRANSPOSE(U_eigvecs_mat)), MATMUL(vx_op, U_eigvecs_mat))
          vy_eig = MATMUL(CONJG(TRANSPOSE(U_eigvecs_mat)), MATMUL(vy_op, U_eigvecs_mat))

          DO n_band = 1, M_dim
             DO m_band = 1, M_dim
             IF (n_band == m_band) CYCLE
             E_n = E_vals(n_band)
             E_m = E_vals(m_band)
             fn = fermi_dist(E_n, 0.0_dp, 1.0_dp)
             fm = fermi_dist(E_m, 0.0_dp, 1.0_dp)
             IF (ABS(fn-fm) < 1.0E-5_dp) CYCLE
             delta_E_real = E_m - E_n 
             denominator_c = CMPLX(delta_E_real, 0.0_dp, dp) * &
                            (CMPLX(delta_E_real - omega, -eta, dp))
             IF (ABS(denominator_c%re) < 1E-12_dp .AND. ABS(denominator_c%im) < 1E-12_dp) CYCLE

             DO alpha = 1, 2 ! 1 for x, 2 for y
                DO beta = 1, 2
                  IF (alpha == 1) THEN
                    v_alpha_comp = vx_eig
                  ELSE
                    v_alpha_comp = vy_eig
                  END IF
                  IF (beta == 1) THEN
                    v_beta_comp = vx_eig
                  ELSE
                    v_beta_comp = vy_eig
                  END IF
                   
                  term_c = (fm-fn) * (v_alpha_comp(n_band, m_band) * v_beta_comp(m_band, n_band)) / denominator_c
                  sigma_accum(ib, alpha, beta) = sigma_accum(ib, alpha, beta) + term_c
                END DO
              END DO
           END DO ! m_band loop
         END DO ! n_band loop
       END DO ! iky loop
       IF (my_rank == 0 .AND. MOD(ikx,MAX(1,Nk/10)) == 0) THEN
           PRINT *, "Progress: ikx = ", ikx, "/", Nk
       ENDIF
     END DO ! ikx loop

    sigma_s_partial = CMPLX(0.0_dp, 0.0_dp, dp)
    sigma_a_partial = CMPLX(0.0_dp, 0.0_dp, dp)
    prefactor_coeff = I_UNIT * e2htoS * (dk_param**2) / (2.0_dp * Pi)

    DO ib = 1, Nbs
       DO alpha = 1, 2
          DO beta = 1, 2
             sigma_s_partial(ib, alpha, beta) = (sigma_accum(ib, alpha, beta) + sigma_accum(ib, beta, alpha)) / 2.0_dp
             sigma_a_partial(ib, alpha, beta) = (sigma_accum(ib, alpha, beta) - sigma_accum(ib, beta, alpha)) / 2.0_dp
             
             sigma_s_partial(ib, alpha, beta) = prefactor_coeff * sigma_s_partial(ib, alpha, beta)
             sigma_a_partial(ib, alpha, beta) = prefactor_coeff * sigma_a_partial(ib, alpha, beta)
          END DO
       END DO
    END DO
    
    CALL mpi_reduce_sum_cmplx(sigma_s_partial, sigma_s_global, Nbs*2*2)
    CALL mpi_reduce_sum_cmplx(sigma_a_partial, sigma_a_global, Nbs*2*2)

    IF (my_rank == 0) THEN
      OPEN(UNIT=10, FILE='S_xx_vs_Bz.dat', STATUS='REPLACE', ACTION='WRITE')
      DO ib = 1, Nbs
        WRITE(10, '(E13.6,A,E13.6,A,E13.6)') Bzs(ib),achar(9), &
                                        sigma_s_global(ib,1,1)%RE, achar(9), sigma_s_global(ib,1,1)%IM
      END DO
      CLOSE(10)

      OPEN(UNIT=11, FILE='S_xy_vs_Bz.dat', STATUS='REPLACE', ACTION='WRITE')
      DO ib = 1, Nbs
        WRITE(11, '(E13.6,A,E13.6,A,E13.6)') Bzs(ib),achar(9), &
                                        sigma_s_global(ib,1,2)%RE, achar(9), sigma_s_global(ib,1,2)%IM
      END DO
      CLOSE(11)

      OPEN(UNIT=12, FILE='A_xy_vs_Bz.dat', STATUS='REPLACE', ACTION='WRITE')
      DO ib = 1, Nbs
        WRITE(12, '(E13.6,A,E13.6,A,E13.6)') Bzs(ib),achar(9), &
                                        sigma_a_global(ib,1,2)%RE, achar(9), sigma_a_global(ib,1,2)%IM
      END DO
      CLOSE(12)
      PRINT *, "Output files generated."
    END IF

    DEALLOCATE(kxs, kys, sigma_accum)
    DEALLOCATE(sigma_s_partial, sigma_a_partial)
    IF (my_rank == 0) THEN
      DEALLOCATE(sigma_s_global, sigma_a_global)
    END IF
    DEALLOCATE(E_vals, U_eigvecs_mat, vx_eig, vy_eig, work, rwork, v_alpha_comp, v_beta_comp)

  END SUBROUTINE calculate_optical_conductivity

END MODULE physics_subroutines

! main_optical_conductivity.f90
PROGRAM main
  USE kinds
  USE constants
  USE physics_subroutines
  USE mpi ! Using an interface module for MPI
  IMPLICIT NONE

  INTEGER :: i,j,nl,Nbs
  REAL(dp),allocatable :: us(:),ms(:,:,:),Bzs(:),mm(:,:)
  REAL(dp) :: omega, eta_val, kf_val, dk_val,Bz

  CALL mpi_setup() ! Initializes MPI, gets rank and size

  ! Parameters from the Python script
  nl = 6
  Nbs = 200
  allocate(us(2*nl),ms(Nbs,nl,3),Bzs(Nbs),mm(Nbs,3*nl))
  open(100,file='ms_vs_Hz_6ls.txt')
    DO i = 1, Nbs
          READ(100, *) Bzs(i),mm(i,:)
    END DO 
    ! 关闭文件
    CLOSE(100)
  do i=1,Nbs
  do j=1,nl
      ms(i,j,:)=mm(i,3*(j-1)+1:3*j)
  enddo
  enddo
  omega = 0.2_dp
  eta_val = 0.01_dp
  kf_val = 2.0_dp
  dk_val = 0.001_dp
  call get_us(nl,us)
  IF (my_rank == 0) THEN
    PRINT *, "Running optical_conductivity with Fortran..."
    PRINT *, "Nl = ", nl
    PRINT *, "omega = ", omega
    PRINT *, "eta = ", eta_val
    PRINT *, "kf = ", kf_val, ", dk = ", dk_val
  END IF

  CALL calculate_optical_conductivity(nl, Nbs,us,Bzs, ms,omega,eta_val, kf_val, dk_val)

  CALL mpi_finalize_custom() ! Finalizes MPI
  deallocate(us,ms,Bzs)

END PROGRAM main
