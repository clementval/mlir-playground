
program main
  implicit none
  integer, parameter :: n = 1024
  real :: x(n), y(n), a = 10.0

  x(:) = 1.0
  y(:) = 2.0


  call saxpy(x,y,n,a)
  print*,sum(y(:))

contains

  subroutine saxpy(x,y,n,a)
    implicit none
    real :: a, x(n), y(n)
    integer :: n, i
    !$ acc kernels
    do i = 1, n
      y(i) = a*x(i)+y(i)
    enddo
    !$ acc end kernels
  end subroutine saxpy

end program
